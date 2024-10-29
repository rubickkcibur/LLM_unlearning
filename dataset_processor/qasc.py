from datasets import load_dataset
import re
COT_EXAMPLES_chat = [
    dict(role="user", content="Question: What type of water formation is formed by clouds?\n(A) pearls\n(B) streams\n(C) shells\n(D) diamonds\n(E) rain\n(F) beads\n(G) cooled\n(H) liquid\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Beads of water are formed by water vapor condensing. Clouds are made of water vapor. Beads of water can be formed by clouds. The answer is (F).\n"),
    dict(role="user", content="Question: Removing what from food will preserve it?\n(A) flavor\n(B) body water\n(C) heat energy\n(D) color\n(E) Water\n(F) Bodily water\n(G) moisture\n(H) ingredients\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Dehydrating food is used for preserving food. Dehydration preserves foods by removing moisture. Removing moisture from food preserves it. The answer is (G).\n"),
    dict(role="user", content="Question: Reproduction is the process by which living things what?\n(A) Most plants\n(B) allow growth\n(C) spread flower seeds\n(D) have wide set eyes\n(E) members of their own species\n(F) have birthing hips\n(G) have quiet laughter\n(H) give birth to babies\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Reproduction is the process by which living things give rise to offspring. Whenever it starts to give birth, it gives birth up to two to four babies offspring. Reproduction is the process by which living things give birth to babies. The answer is (H).\n"),
]

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

def train_data():
    train_data = []
    data = load_dataset("allenai/qasc", split="train[30:]")
    for d in data:
        train_data.append(
            [
                {"role": "user", "content": "Question: {}\n{}\nLet's think step by step:\n".format(
                    d["question"],
                    "\n".join(["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
                )},
                {"role": "assistant", "content": "Answer: {}. The answer is ({}).\n".format(
                    " ".join([d["fact1"], d["fact2"], d["combinedfact"]]),
                    d["answerKey"]
                )}
            ]
        )
    return train_data

def self_data():
    data = load_dataset("allenai/qasc", split="train[30:]")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content="Question: {}\n{}\nLet's think step by step:\n".format(
            d["question"],
            "\n".join(
                ["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
        ))]
        for d in data
    ]
    questions = [
        "Question: {}\n{}\nLet's think step by step:\n".format(
            d["question"],
            "\n".join(
                ["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
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
    data = load_dataset("allenai/qasc", split="validation")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content = "Question: {}\n{}\nLet's think step by step:\n".format(
                    d["question"],
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
