from dataset_processor import gsm8k, medmcqa, qasc

COT_CHAT = {}
COT_CHAT["gsm8k"] = gsm8k.COT_EXAMPLES_chat
COT_CHAT["medmcqa"] = medmcqa.COT_EXAMPLES_chat
COT_CHAT["qasc"] = qasc.COT_EXAMPLES_chat

COT_base = {}
COT_base["gsm8k"] = gsm8k.COT_EXAMPLES_base
COT_base["medmcqa"] = medmcqa.COT_EXAMPLES_base
COT_base["qasc"] = qasc.COT_EXAMPLES_base

TRAIN_DATA = {}
TRAIN_DATA["gsm8k"] = gsm8k.train_data
TRAIN_DATA["medmcqa"] = medmcqa.train_data
TRAIN_DATA["qasc"] = qasc.train_data

SELF_DATA = {}
SELF_DATA["gsm8k"] = gsm8k.self_data
SELF_DATA["medmcqa"] = medmcqa.self_data
SELF_DATA["qasc"] = qasc.self_data

TEST_DATA = {}
TEST_DATA["gsm8k"] = gsm8k.test_data
TEST_DATA["medmcqa"] = medmcqa.test_data
TEST_DATA["qasc"] = qasc.test_data

METRIC = {}
METRIC["gsm8k"] = gsm8k.metric
METRIC["medmcqa"] = medmcqa.metric
METRIC["qasc"] = qasc.metric
# import datasets
# dataset = datasets.load_dataset("openlifescienceai/medmcqa",split="validation")
# for d in dataset:
#     print(d["cop"])
#     quit()