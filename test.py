import datasets
import transformers
import torch
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/aifs4su/rubickjiang/huggingface_models/Meta-Llama-3-8B",
    model_max_length=16,
    padding_side="left",
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

a = " ".join(["I am your father!", "I am your gradfather!"])
b = " ".join(["I am your see you tomorrow", "I am your gradfather!"])
prompts = [a, b]
encoding = tokenizer(
    prompts,
    add_special_tokens=True,
    max_length=16,
    truncation=True,
    padding="max_length",
    return_tensors='pt'
)
print(encoding)