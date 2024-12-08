from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np

from trainLM_tools import SupervisedDataset, build_model, MyTrainer
from datasets import load_dataset
import os
from accelerate.logging import get_logger
import logging
from dataset_processor.processor_registers import *

m = transformers.LlamaForCausalLM
'''
this script is to train LLM on filtered generated data. The two major usage is:
1. train baseline methods
2. train modelL in ADADF
'''

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/incoming/LLM/llama2/llama2-7b")
    filter_base_model_path: str = field(default="")
    peft_model_path: str = field(default="")


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
    temp_data_path: str = field(
        default=None
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
    special_weight_path: str = field(
        default="", metadata={"help": "special weight"}
    )
    unlearning_alpha: float = field(
        default=0.2
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    vocab_size: int = field(default=None)
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


def load_valid_data(valid_data_path):
    assert ":" in valid_data_path
    name, split = valid_data_path.split(":")
    if split.isdigit():
        dataset = load_dataset(name, data_dir="main", split="train[{}:]".format(split)) if name in [
            "gsm8k"] else load_dataset(name, split="train[{}:]".format(split))
    else:
        dataset = load_dataset(name, split=split)
    print("Load valid dataset, total length is {}".format(len(dataset)))
    return dataset



def load_train_data(tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, is_chat_model=True):
    train_data = TRAIN_DATA[data_args.dataset_name]()
    # data = load_dataset("gsm8k", data_dir="main", split="train")
    # for d in data:
    #     train_data.append(
    #         [
    #             {"role": "user", "content": "Question: {}\nLet's think step by step:\n".format(d["question"])},
    #             {"role": "assistant", "content": "Answer: {}\n".format(format_ground_truth_answer(d["answer"]))}
    #         ]
    #     )
    train_dataset = SupervisedDataset(
        train_data,
        tokenizer=tokenizer,
        max_len=max_len,
        is_chat_model=is_chat_model
    )
    return train_dataset


def baseline():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    logger.info('Initializing model...')

    modelL, tokenizerL = build_model(model_args, training_args, lora_args, logger)
    tokenizerL.pad_token = tokenizerL.eos_token
    tokenizerL.pad_token_id = tokenizerL.eos_token_id
    training_args.vocab_size = modelL.config.vocab_size
    modelL.to(device)

    logger.info('Loading training&evaluation data...')
    is_chat_model = "Instruct" in model_args.model_name_or_path
    train_dataset = load_train_data(tokenizerL, data_args, training_args.model_max_length, is_chat_model)
    trainer = Trainer(
        modelL,
        training_args,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    seed = 114514
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    baseline()


