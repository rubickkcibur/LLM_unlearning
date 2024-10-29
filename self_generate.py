from dataclasses import dataclass, field
from typing import Dict, Optional, List

import jsonlines
import transformers
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
import logging
import tqdm
from peft import PeftModel
from dataset_processor.processor_registers import *

'''
This script is to evaluate the LLM's performance on test dataset
'''

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/incoming/LLM/llama2/llama2-7b")
    filter_base_model_path: str = field(default="")
    vocab_size: int = field(default=0)
    peft_model_path: str = field(default="")
    mode: str = field(default="chat")


@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/LAB/jiangcy/AdaDF/samples/gsm8k_test.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    valid_data_path: str = field(
        default=None, metadata={"help": "valid data path, name:split"}
    )
    dataset_name: str = field(
        default=None
    )
    data_filter_mode: str = field(
        default="Consistency", metadata={"help": "Consistency, Groundtruth, Entropy, Weighted"}
    )
    lazy_preprocess: bool = False
    uncertainty_th: float = field(
        default=1.0
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=800,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    filter_training_batch_size: int = field(default=8)
    valid_batch_size: int = field(default=16)
    filter_training_epochs: int = field(default=10)
    filter_model_lr: float = field(
        default=1e-3
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def evaluation_main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # output_dir = "modelL-filter_strategy_{}-time_{}".format(data_args.data_filter_mode, int(time.time()))
    # training_args.output_dir = os.path.join(training_args.output_dir, output_dir)
    # os.makedirs(training_args.output_dir, exist_ok=True)
    # ROLE_CONTENT = "You are a calculation assistant. You will be given an arithmetic question. Please think step by step and give the answer. After giving your thoughts, use 'The answer is:' followed by the answer."
    accelerator = Accelerator()
    device = accelerator.device

    logger.info('Loading causal model...')
    modelL = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    if len(model_args.peft_model_path) > 0:
        logger.info("loading peft weights from{}".format(model_args.peft_model_path))
        modelL = PeftModel.from_pretrained(modelL, model_args.peft_model_path)
        modelL.merge_and_unload()
    tokenizerL = transformers.AutoTokenizer.from_pretrained(
        "/aifs4su/rubickjiang/huggingface_models/Meta-Llama-3-8B-Instruct" 
            if model_args.mode == "chat" 
            else "/aifs4su/rubickjiang/huggingface_models/Meta-Llama-3-8B",
        model_max_length=training_args.model_max_length,
        use_fast=False,
        padding_side="left")
    tokenizerL.pad_token_id = tokenizerL.eos_token_id
    
    terminators = [
        tokenizerL.eos_token_id,
        tokenizerL.convert_tokens_to_ids("<|eot_id|>")
    ] if model_args.mode == "chat" else tokenizerL.eos_token_id

    prompts_dataset = SELF_DATA[data_args.dataset_name]()
    answers = prompts_dataset["ground"]
    questions = prompts_dataset["questions"]
    prompts = prompts_dataset["inputs"]

    def prepare_prompts(prompts, tokenizer, model_args, batch_size=16):
        if model_args.mode == "chat":
            inputs = [
                tokenizerL.apply_chat_template(
                    m,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False
                )
                for m in prompts
            ]
        else:
            inputs = [
                "".join([x["content"] for x in rounds]) + "Answer:"
                for rounds in prompts
            ]
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        batches_tok = []
        tokenizer.padding_side = "left"
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=training_args.model_max_length,
                    add_special_tokens=True).to(device)
            )
        return batches_tok

    modelL.eval()
    modelL.to(device)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(prompts) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts, tokenizerL, model_args, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            with torch.no_grad():
                outputs_tokenized = modelL.generate(
                    **prompts_tokenized,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizerL.eos_token_id,
                    # do_sample=True
                )

            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):]
                                 for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]

            # count and decode gen. tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizerL.batch_decode(outputs_tokenized)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            if accelerator.is_local_main_process:
                pbar.update(1)
            torch.cuda.empty_cache()
        results = [results]  # transform to list, otherwise gather_object() will not collect correctly
    results_gathered = gather_object(results)
    if accelerator.is_main_process:
        dump_objs = []
        total_results = []
        for r in results_gathered:
            total_results += r["outputs"]

        for i in range(len(total_results)):
            pred_sentence = total_results[i]
            answer = answers[i]
            acc = METRIC[data_args.dataset_name]([pred_sentence], [answer])
            if acc <= 0:
                dump_objs.append({
                    "user": questions[i],
                    "assistant": pred_sentence.replace("<|eot_id|>", "")
                })
        dump_path = os.path.join("/aifs4su/rubickjiang/unlearning/data/self_generated_base/", data_args.dataset_name)
        os.makedirs(dump_path, exist_ok=True)
        with jsonlines.open(os.path.join(dump_path, "wrong_answer.jsonl"), "w") as writer:
            for obj in dump_objs:
                writer.write(obj)


if __name__ == "__main__":
    # 注意seed，原设置是没有do_sample的
    seed = os.environ.get("SEED", 114514)
    seed = int(seed)
    print("================set global random seed to {}================".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    evaluation_main()


