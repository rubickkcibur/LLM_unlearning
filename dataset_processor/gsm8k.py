from datasets import load_dataset
import re
COT_EXAMPLES_chat = [
    dict(role="user", content="Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n"),
    dict(role="user", content="Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n"),
    dict(role="user", content="Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n"),
]

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

def train_data():
    def format_answer(ground: str):
        ground = ground.replace("####", "The answer is")
        ground = re.sub(r"<<.*?>>", "", ground)
        return ground
    train_data = []
    data = load_dataset("gsm8k", data_dir="main", split="train")
    for d in data:
        train_data.append(
            [
                {"role": "user", "content": "Question: {}\nLet's think step by step:\n".format(d["question"])},
                {"role": "assistant", "content": "Answer: {}\n".format(format_answer(d["answer"]))}
            ]
        )
    return train_data

def self_data():
    def derive_num_from_answer(answer_test):
        num = answer_test.split("####")[-1]
        num = num.strip()
        num = num.replace(",", "")
        return num
    data = load_dataset("gsm8k", data_dir="main", split="train")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content = "Question: {}\nLet's think step by step.\n".format(d["question"]))]
        for d in data
    ]
    questions = [
        "Question: {}\nLet's think step by step.\n".format(d["question"])
        for d in data
    ]
    ground = [
        derive_num_from_answer(d["answer"])
        for d in data
    ]
    return {
        "inputs": inputs,
        "questions": questions,
        "ground": ground
    }

def test_data():
    def derive_num_from_answer(answer_test):
        num = answer_test.split("####")[-1]
        num = num.strip()
        num = num.replace(",", "")
        return num
    data = load_dataset("gsm8k", data_dir="main", split="test")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content = "Question: {}\nLet's think step by step.\n".format(d["question"]))]
        for d in data
    ]
    ground = [
        derive_num_from_answer(d["answer"])
        for d in data
    ]
    return {
        "inputs": inputs,
        "ground": ground
    }

def metric(output_text, ground):
    def derive_num_from_output(output_text):
        new_text = output_text.lower()
        new_text = new_text.split("question:")[0]
        suffix = new_text.split("the answer is")[-1]
        suffix = suffix.strip()
        if "=" in suffix:
            suffix = suffix.split("=")[-1].strip()
        if len(suffix) <= 0:
            return None
        pattern = r"(\D*?)(\d+\.?\d*)"
        ret = re.search(pattern, suffix.replace(",", ""))
        if ret is None:
            return None
        num = ret.group(2)
        if len(num) > 40:
            return None
        return num
    preds = [
        derive_num_from_output(txt)
        for txt in output_text
    ]
    assert len(preds) == len(ground)
    acc = [
        1 if (preds[i] is not None and int(float(preds[i])) == int(float(ground[i]))) else 0
        for i in range(len(preds))
    ]
    return sum(acc)/len(acc)
