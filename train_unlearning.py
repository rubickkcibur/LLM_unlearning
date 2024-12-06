import jsonlines
from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np

from trainLM_tools import SupervisedDataset, build_model, MyTrainer
from datasets import load_dataset, DatasetDict
import os
from accelerate.logging import get_logger
import logging
from dataset_processor.processor_registers import *
import copy
import wandb

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
    unlearning_portion: float = field(
        default=0.25
    )
    arrange: str = field(
        default="interpolation", metadata={"help": "interpolation, ahead"}
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
    unlearning_alpha: float = field(
        default=0.2
    )
    seed: int = field(
        default = 114514
    )
    data_seed: int = field(
        default = 114514
    )
    unlearning_loss: str = field(
        default="GA"
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


def load_valid_data(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args, logger, is_chat_model = True):
    logger.info("loading qasc valid data.........")
    qasc_data = TEST_DATA["qasc"]()["inputs"]
    qasc_data = qasc_data[:512]
    qasc_dataset = SupervisedDataset(
        qasc_data,
        tokenizer=tokenizer,
        max_len=training_args.model_max_length,
        is_chat_model = is_chat_model
    )
    logger.info("loading gsm8k valid data.........")
    gsm8k_data = TEST_DATA["gsm8k"]()["inputs"]
    gsm8k_data = gsm8k_data[:512]
    gsm8k_dataset = SupervisedDataset(
        gsm8k_data,
        tokenizer=tokenizer,
        max_len=training_args.model_max_length,
        is_chat_model = is_chat_model
    )
    logger.info("loading medmcqa valid data.........")
    medmcqa_data = TEST_DATA["medmcqa"]()["inputs"]
    medmcqa_data = medmcqa_data[:512]
    medmcqa_dataset = SupervisedDataset(
        medmcqa_data,
        tokenizer=tokenizer,
        max_len=training_args.model_max_length,
        is_chat_model = is_chat_model
    )
    logger.info("loading aqua valid data.........")
    aqua_data = TEST_DATA["aqua"]()["inputs"]
    aqua_data = aqua_data[:512]
    aqua_dataset = SupervisedDataset(
        aqua_data,
        tokenizer=tokenizer,
        max_len=training_args.model_max_length,
        is_chat_model = is_chat_model
    )
    return DatasetDict({"qasc_valid": qasc_dataset, "gsm8k_valid": gsm8k_dataset, "medmcqa_valid": medmcqa_dataset, "aqua_valid": aqua_dataset})



def load_train_data(tokenizer: transformers.PreTrainedTokenizer, data_args, max_len):
    train_data = TRAIN_DATA[data_args.dataset_name]()
    train_dataset = SupervisedDataset(
        train_data,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return train_dataset

def load_unlearning_data(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args, logger, is_chat_model = True):
    max_len = training_args.model_max_length
    unlearning_portion = data_args.unlearning_portion
    unlearning_alpha = training_args.unlearning_alpha

    if ";" in data_args.dataset_name:
        train_data = []
        names = data_args.dataset_name.split(";")
        for name in names:
            train_data.extend(TRAIN_DATA[name]())
    else:
        train_data = TRAIN_DATA[data_args.dataset_name]()
    neg_files = data_args.data_path.strip().split(";")
    neg_samples = []
    for file in neg_files:
        logger.info("load neg data from {}........".format(file))
        with jsonlines.open(file, "r") as reader:
            for obj in reader:
                neg_samples.append(
                    [
                        {"role": "user", "content": obj["user"]},
                        {"role": "assistant", "content": obj["assistant"]}
                    ]
                )
    max_neg_n = int(len(train_data) * unlearning_portion)
    total_neg_num = min(len(neg_samples), max_neg_n)
    logger.info("pos data number is {}, neg data number is {}".format(len(train_data), total_neg_num))
    if total_neg_num <= 0:
        total_data = copy.deepcopy(train_data)
        weights = [1] * len(train_data)
    elif data_args.arrange == "ahead":
        total_data = copy.deepcopy(neg_samples[:total_neg_num])
        weights = [-unlearning_alpha] * total_neg_num
        total_data.extend(copy.deepcopy(train_data))
        weights.extend([1]*len(train_data))
    else:
        round_n = int(1 / unlearning_portion)
        logger.info("round number is {}".format(round_n))
        total_data = []
        weights = []
        neg_p = 0
        for i in range(len(train_data)):
            if i % (round_n - 1) == 0 and neg_p < total_neg_num:
                total_data.append(neg_samples[neg_p])
                neg_p += 1
                weights.append(-unlearning_alpha)
            total_data.append(train_data[i])
            weights.append(1)

    # total_data = copy.deepcopy(train_data)
    # if max_neg_n > 0:
    #     total_data += copy.deepcopy(neg_samples[:max_neg_n])
    # weights = [1] * len(train_data)
    # if max_neg_n > 0:
    #     weights += [-unlearning_alpha] * min(len(neg_samples), max_neg_n)
        
    train_dataset = SupervisedDataset(
        total_data,
        tokenizer=tokenizer,
        max_len=max_len,
        weights=weights,
        is_chat_model = is_chat_model
    )
    return train_dataset


def baseline_ds():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    parsed_args = parser.parse_args_into_dataclasses()
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parsed_args

    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    with open(os.path.join(training_args.output_dir, "args.log"), "w") as f:
        f.write(str(parsed_args))

    logger.info('Initializing model...')
    #todo: 在output_dir存一个args的file
    modelL, tokenizerL = build_model(model_args, training_args, lora_args, logger)
    tokenizerL.pad_token = tokenizerL.eos_token
    tokenizerL.pad_token_id = tokenizerL.eos_token_id
    training_args.vocab_size = modelL.config.vocab_size
    modelL.to(device)

    logger.info('Loading unlearning data...')

    train_dataset = load_unlearning_data(tokenizerL, data_args, training_args, logger, is_chat_model = ("Instruct" in model_args.model_name_or_path))
    valid_datasets = load_valid_data(tokenizerL, data_args, training_args, logger, is_chat_model = ("Instruct" in model_args.model_name_or_path))
    trainer = MyTrainer(
        modelL,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_datasets
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    seed = 114514
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["WANDB_PROJECT"]="unlearning"
    os.environ["WANDB_DIR"]="/aifs4su/rubickjiang/wandb"
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_WATCH"]="false"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    #Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    baseline_ds()


