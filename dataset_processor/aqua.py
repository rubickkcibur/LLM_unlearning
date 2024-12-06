from datasets import load_dataset
import re
COT_EXAMPLES_chat = [
    dict(role="user", content="Question: Three birds are flying at a fast rate of 900 kilometers per hour. What is their speed in miles per minute? [1km = 0.6 miles]\n(A) 32400\n(B) 6000\n(C) 600\n(D) 60000\n(E) 9\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: First, convert kilometers to miles. The speed is 900 kilometers per hour. 1 kilometer is equal to 0.6 miles. So, 900 kilometers is 900*0.6=540 miles. Second, Convert hours to minutes. There are 60 minutes in one hour. We need to convert 540 miles per hour to miles per minute. So, 540/60=9 miles per minute. The answer is (E).\n"),
    dict(role="user", content="Question: If a number of apples, say x, is divisible by 3 and 9, x is a multiple of which of the following values?\n(A) 15\n(B) 27\n(C) 14\n(D) 12\n(E) 20\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: If a number x is divisible by both 3 and 9, then x must be a multiple of the least common multiple (LCM) of these two numbers. Since 9 is a multiple of 3, the LCM of 3 and 9 is 9 itself. 15, 14, 12, and 20 are not the multiple of 9. The answer is (B).\n"),
    dict(role="user", content="Question: A man earns RS. 20 on the first day and spends Rs. 15 on the next day. He again earns Rs. 20 on the third day and spends Rs. 15 on the fourth day. If he continues to save like this, how soon will he have Rs. 60 in hand?\n(A) on the 17th day\n(B) on the 27th day\n(C) on the 30th day\n(D) on the 40th day\n(E) on the 44th day\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: On the 1st day, he earns Rs. 20. On the 2nd day, he spends Rs. 15. This pattern repeats every two days. So, over a span of two days, his net gain is 20-15=5 Rs. On the last day he reaches Rs. 60, he will earn 20 and have a 60-20=40 Rs deposit. So the total time he spends to earn 60 is (40/5)*2 + 1 = 17 days. The answer is (A).\n"),
]

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

def train_data():
    pass
def self_data():
    data = load_dataset("/aifs4su/rubickjiang/public_data/aqua_rat", "raw", split="train")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content="Question: {}\n{}\nLet's think step by step:\n".format(
            d["question"],
            "\n".join(
                ["(" + option.replace(")", ") ") for option in d["options"]])
        ))]
        for d in data
    ]
    questions = [
        "Question: {}\n{}\nLet's think step by step:\n".format(
            d["question"],
            "\n".join(
                ["(" + option.replace(")", ") ") for option in d["options"]])
        )
        for d in data
    ]
    ground = [
        d["correct"]
        for d in data
    ]
    return {
        "inputs": inputs,
        "questions": questions,
        "ground": ground
    }

def test_data():
    data = load_dataset("/aifs4su/rubickjiang/public_data/aqua_rat", "raw", split="test")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content = "Question: {}\n{}\nLet's think step by step:\n".format(
            d["question"],
            "\n".join(
                ["(" + option.replace(")", ") ") for option in d["options"]])
        ))]
        for d in data
    ]
    ground = [
        d["correct"]
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
