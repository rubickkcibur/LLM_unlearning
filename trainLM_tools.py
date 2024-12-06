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
import datasets
from transformers.trainer_utils import seed_worker
import torch.nn.functional as F
import math

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
        if "weights" not in inputs:
            batch_size = logits.shape[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.args.vocab_size)
            shift_labels = shift_labels.view(-1)
            # loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = loss.view(batch_size, -1)
            loss = torch.mean(loss)
            return (loss, outputs) if return_outputs else loss
        
        weights = inputs["weights"]
        # print(torch.sum(weights))
        if self.args.unlearning_loss == "GA":
            #gradient ascent
            batch_size = logits.shape[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.args.vocab_size)
            shift_labels = shift_labels.view(-1)
            # loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = loss.view(batch_size, -1)
            loss = torch.mul(weights.unsqueeze(-1), loss)
            loss = torch.mean(loss)
        elif self.args.unlearning_loss == "ME":
            #max entropy
            pos_weights = torch.where(weights > 0, weights, 0)
            neg_weights = torch.where(weights <= 0, -weights, 0)
            batch_size = logits.shape[0]
            #compute pos loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.args.vocab_size)
            shift_labels = shift_labels.view(-1)
            # loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            pos_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            pos_loss = pos_loss.view(batch_size, -1)
            pos_loss = torch.mul(pos_weights.unsqueeze(-1), pos_loss)
            #compute neg loss
            neg_shifted_logits = logits[..., :-1, :].contiguous()
            log_softmax_logits = F.log_softmax(neg_shifted_logits, dim = -1)
            softmax_logits = F.softmax(neg_shifted_logits, dim = -1)
            entropy = - (softmax_logits * log_softmax_logits).sum(dim = -1)
            max_entropy = math.log(self.args.vocab_size)
            neg_loss = 0.5 * (max_entropy - entropy) ** 2
            neg_loss = neg_loss.view(batch_size, -1)
            neg_loss = torch.mul(neg_weights.unsqueeze(-1), neg_loss)
            # add
            loss = pos_loss + neg_loss
            loss = torch.mean(loss)
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self) -> DataLoader:
        """
        My dataloader: copied from Trainer, shuffle is False
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

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

