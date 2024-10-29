from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import jsonlines
import copy
# from tqdm.auto import tqdm
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import PeftModel

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.get("logits")
        labels = inputs["labels"]
        weights = inputs["weights"]
        batch_size = logits.shape[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.args.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(batch_size, -1)
        loss = torch.mul(weights.unsqueeze(-1), loss)
        loss = torch.mean(loss)
        return (loss, outputs) if return_outputs else loss

# def maybe_zero_3(param):
#     if hasattr(param, "ds_id"):
#         assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param


# Borrowed from peft.utils.get_peft_model_state_dict
# def get_peft_state_maybe_zero_3(named_params, bias):
#     if bias == "none":
#         to_return = {k: t for k, t in named_params if "lora_" in k}
#     elif bias == "all":
#         to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
#     elif bias == "lora_only":
#         to_return = {}
#         maybe_lora_bias = {}
#         lora_bias_names = set()
#         for k, t in named_params:
#             if "lora_" in k:
#                 to_return[k] = t
#                 bias_name = k.split("lora_")[0] + "bias"
#                 lora_bias_names.add(bias_name)
#             elif "bias" in k:
#                 maybe_lora_bias[k] = t
#         for k, t in maybe_lora_bias:
#             if bias_name in lora_bias_names:
#                 to_return[bias_name] = t
#     else:
#         raise NotImplementedError
#     to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
#     return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
#     """Collects the state dict and dump to disk."""
#     # check if zero3 mode enabled
#     if deepspeed.is_deepspeed_zero3_enabled():
#         state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
#     else:
#         if trainer.args.use_lora:
#             state_dict = get_peft_state_maybe_zero_3(
#                 trainer.model.named_parameters(), bias
#             )
#         else:
#             state_dict = trainer.model.state_dict()
#     if trainer.args.should_save and trainer.args.local_rank == 0:
#         trainer._save(output_dir, state_dict=state_dict)


def qa_preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        processor,
        system_message: str = ""
) -> Dict:
    input_ids = []
    targets = []
    masks = []
    # input_text = ["Q: " + source["question"] + tokenizer.eos_token + "A: " + source["answer"] for source in sources]
    input_text = [processor(source) for source in sources]
    encoding = tokenizer(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"]
    targets = copy.deepcopy(encoding["input_ids"])
    masks = encoding['attention_mask']
    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
        attention_mask=masks,
    )


# def preprocess(
#         sources,
#         tokenizer: transformers.PreTrainedTokenizer,
#         max_len: int,
#         system_message: str = "You are a helpful assistant."
# ) -> Dict:
#     roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
#
#     im_start = tokenizer.im_start_id
#     im_end = tokenizer.im_end_id
#     nl_tokens = tokenizer('\n').input_ids
#     _system = tokenizer('system').input_ids + nl_tokens
#     _user = tokenizer('user').input_ids + nl_tokens
#     _assistant = tokenizer('assistant').input_ids + nl_tokens
#
#     # Apply prompt templates
#     input_ids, targets = [], []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != roles["user"]:
#             source = source[1:]
#
#         input_id, target = [], []
#         system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
#         input_id += system
#         target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
#         assert len(input_id) == len(target)
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             _input_id = tokenizer(role).input_ids + nl_tokens + \
#                         tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
#             input_id += _input_id
#             if role == '<|im_start|>user':
#                 _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
#             elif role == '<|im_start|>assistant':
#                 _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
#                           _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
#             else:
#                 raise NotImplementedError
#             target += _target
#         assert len(input_id) == len(target)
#         input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
#         target += [IGNORE_TOKEN_ID] * (max_len - len(target))
#         input_ids.append(input_id[:max_len])
#         targets.append(target[:max_len])
#     input_ids = torch.tensor(input_ids, dtype=torch.int)
#     targets = torch.tensor(targets, dtype=torch.int)
#
#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, chat_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, weights = None, is_chat_model = True):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        if is_chat_model:
            self.sources = [
                tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)
                for x in chat_data
            ]
        else:
            self.sources = [
                "".join([x[0]["content"], x[1]["content"]])
                for x in chat_data
            ]
        encoding = tokenizer(
            self.sources,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )
        self.input_ids = encoding["input_ids"]
        self.labels = copy.deepcopy(encoding["input_ids"])
        self.attention_mask = encoding['attention_mask']
        self.weights = weights

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.weights is not None:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                attention_mask=self.attention_mask[i],
                weights=self.weights[i]
            )
        else:
            return dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                attention_mask=self.attention_mask[i],
            )


# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""
#
#     def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
#         super(LazySupervisedDataset, self).__init__()
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.raw_data = raw_data
#         self.cached_data_dict = {}
#
#     def __len__(self):
#         return len(self.raw_data)
#
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         if i in self.cached_data_dict:
#             return self.cached_data_dict[i]
#
#         ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
#         ret = dict(
#             input_ids=ret["input_ids"][0],
#             labels=ret["labels"][0],
#             attention_mask=ret["attention_mask"][0],
#         )
#         self.cached_data_dict[i] = ret
#
#         return ret


# def make_supervised_data_module(
#         tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, weights
# ) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     dataset_cls = (
#         LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
#     )
#     rank0_print("Loading data...")
#     train_data = []
#     with jsonlines.open(data_args.data_path, "r") as reader:
#         for idx, obj in enumerate(reader):
#             question = obj["question"]
#             candidates = obj["candidates"]
#             cands_weight = weights[idx]
#             assert len(candidates) == len(cands_weight)
#             for i in range(len(candidates)):
#                 train_data.append({
#                     "question": question,
#                     "answer": candidates[i],
#                     "weight": cands_weight[i]
#                 })
#     # train_json = json.load(open(data_args.data_path, "r"))
#     train_dataset = dataset_cls(
#         train_data,
#         tokenizer=tokenizer,
#         max_len=max_len,
#         data_processor=lambda x: "Q: " + x["question"] + tokenizer.eos_token + "A: " + x["answer"]
#     )
#
#     # if data_args.eval_data_path:
#     #     eval_json = json.load(open(data_args.eval_data_path, "r"))
#     #     eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
#     # else:
#     #     eval_dataset = None
#     eval_dataset = None
#
#     return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def build_model(model_args, training_args, lora_args, logger):
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # gptq_config = GPTQConfig(
    #     bits=8, disable_exllama=True, tokenizer=tokenizer, dataset="c4"
    # )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        # quantization_config=gptq_config,
        torch_dtype=torch.bfloat16,
    )

    if len(model_args.peft_model_path) > 0:
        logger.info('loading peft model from {}'.format(model_args.peft_model_path))
        model = PeftModel.from_pretrained(model, model_args.peft_model_path)
        model.merge_and_unload()
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        return model, tokenizer
    if training_args.use_lora:
        if "Instruct" in model_args.model_name_or_path:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        logger.info("use peft!")
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            # target_modules=lora_args.lora_target_modules,
            target_modules = ["q_proj","k_proj","v_proj","o_proj","down_proj","gate_proj","up_proj"],
            # target_modules=["q_proj", "v_proj", "down_proj", "gate_proj", "up_proj"],
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )

        # model = prepare_model_for_kbit_training(
        #     model, use_gradient_checkpointing=training_args.gradient_checkpointing
        # ) #q_lora
        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            # model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    return model, tokenizer


# def trainL(training_args, lora_args, model_args, model, tokenizer, train_dataset, eval_dataset=None):
#     global local_rank
#     logger = get_logger(__name__)
#     num_training_steps = training_args.num_train_epochs * len(train_dataset)
#     num_training_steps = int(num_training_steps)
#     accelerate = Accelerator()
#     device = accelerate.device
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
#     lr_scheduler = get_scheduler(
#         training_args.lr_scheduler_type,
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=num_training_steps,
#     )
#     # my train loop
#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=training_args.per_device_train_batch_size,
#     )
#     model, train_dataloader, optimizer, lr_scheduler = accelerate.prepare(
#         model, train_dataloader, optimizer, lr_scheduler
#     )
#
#     pbar = tqdm.tqdm(total=len(train_dataloader) * int(training_args.num_train_epochs),
#                      disable=(not accelerate.is_local_main_process))
#     print(training_args.num_train_epochs)
#     for epoch in range(int(training_args.num_train_epochs)):
#         model.train()
#         total_loss = []
#         for batch in train_dataloader:
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 labels=batch["labels"],
#                 attention_mask=batch["attention_mask"],
#             )
#             logits = outputs.get("logits")
#             labels = batch["labels"]
#             weights = batch["weight"]
#             # Shift so that tokens < n predict n
#             batch_size = logits.shape[0]
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#             shift_logits = shift_logits.view(-1, model_args.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             # shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)
#             loss = loss.view(batch_size, -1)
#             loss = torch.mul(weights.unsqueeze(-1), loss)
#             loss = torch.mean(loss)
#             accelerate.backward(loss)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             if accelerate.sync_gradients:
#                 pbar.update(batch_size)
#             total_loss.append(loss.detach().item())
#         accelerate.print("total loss: {}".format(sum(total_loss) / len(total_loss) if len(total_loss) > 0 else 0))
#     accelerate.wait_for_everyone()
#     accelerate.end_training()
