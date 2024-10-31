from datasets import load_dataset
import re
COT_EXAMPLES_chat = [
    dict(role="user", content="Question: Poison causes harm to which of the following?\n(A) a Tree\n(B) a robot\n(C) a house \n(D)a car\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Poison will harm living things, only a tree is a living thing. The answer is (A).\n"),
    dict(role="user", content="Question: As you look deeper into a Marbel you can see?\n(A) the future\n(B) minut defects\n(C) colors\n(D) the other side\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects. The answer is (B)\n"),
    dict(role="user", content="Question: When food is reduced in the stomach?\n(A) the mind needs time to digest\n(B) take a second to digest what I said\n(C) nutrients are being deconstructed\n(D) reader’s digest is a body of works\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: The food is being deconstructed in the stomach during digestion. The answer is (C).\n"),
]

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

def train_data():
    train_data = []
    data = load_dataset("/aifs4su/rubickjiang/public_data/openbookqa", "additional", split="train")
    for d in data:
        train_data.append(
            [
                {"role": "user", "content": "Question: {}?\n{}\nLet's think step by step:\n".format(
                    d["question_stem"],
                    "\n".join(["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
                )},
                {"role": "assistant", "content": "Answer: {}. The answer is ({}).\n".format(
                    d["fact1"],
                    d["answerKey"]
                )}
            ]
        )
    return train_data

def self_data():
    data = load_dataset("/aifs4su/rubickjiang/public_data/openbookqa", "additional", split="train")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content="Question: {}?\n{}\nLet's think step by step:\n".format(
            d["question_stem"],
            "\n".join(["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
        ))]
        for d in data
    ]
    questions = [
        "Question: {}?\n{}\nLet's think step by step:\n".format(
            d["question_stem"],
            "\n".join(["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
        )
        for d in data
    ]
    ground = [
        d["answerKey"]
        for d in data
    ]
    return {
        "inputs": inputs,
        "questions": questions,
        "ground": ground
    }

def test_data():
    data = load_dataset("/aifs4su/rubickjiang/public_data/openbookqa", "additional", split="test")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content = "Question: {}?\n{}\nLet's think step by step:\n".format(
            d["question_stem"],
            "\n".join(["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
        ))]
        for d in data
    ]
    ground = [
        d["answerKey"]
        for d in data
    ]
    return {
        "inputs": inputs,
        "ground": ground
    }

def metric(output_text, ground):
    def derive_choice_from_output(output_text):
        new_text = output_text.lower()
        new_text = new_text.split("question:")[0]
        suffix = new_text.split("the answer is")
        if len(suffix) > 2:
            suffix = suffix[1]
        else:
            suffix = suffix[-1]
        suffix = suffix.strip()
        if "=" in suffix:
            suffix = suffix.split("=")[-1].strip()
        if len(suffix) <= 0:
            return None
        pattern = r"\(([a-z])\)"
        ret = re.search(pattern, suffix.replace(",", ""))
        if ret is None:
            return None
        choice = ret.group(1)
        if len(choice) > 30:
            return None
        return choice.strip().upper()
    preds = [
        derive_choice_from_output(txt)
        for txt in output_text
    ]
    assert len(preds) == len(ground)
    acc = [
        1 if (preds[i] is not None and preds[i] == ground[i]) else 0
        for i in range(len(preds))
    ]
    return sum(acc)/len(acc)
