import os
import jsonlines
objs = []
from datasets import load_dataset
dataset = load_dataset("/aifs4su/rubickjiang/public_data/qasc", split="train[30:]")
questions = []
refined_data = []
with jsonlines.open("/aifs4su/rubickjiang/unlearning/data/self_generated_base/qasc/wrong_answer.jsonl", "r") as reader:
    for obj in reader:
        q = obj["user"].strip()
        q = q.split("\n")[0]
        q = q.replace("Question: ", "")
        questions.append(q)
        refined_data.append({
            "user": obj["user"],
            "assistant": "Answer:" + obj["assistant"].split("Question:")[0]
        })
ground_data = []
for d in dataset:
    d_q = d["question"]
    if d_q in questions:
        ground_data.append({
            "user": "Question: {}\n{}\nLet's think step by step:\n".format(
                    d["question"],
                    "\n".join(["({}) {}".format(label, text) for label, text in zip(d["choices"]["label"], d["choices"]["text"])])
                ),
            "assistant": "Answer: {}. The answer is ({}).\n".format(
                    " ".join([d["fact1"], d["fact2"], d["combinedfact"]]),
                    d["answerKey"]
                )
        })
with jsonlines.open("/aifs4su/rubickjiang/unlearning/data/self_generated_base/qasc/wrong_answer_shorter.jsonl", "w") as writer:
    for d in refined_data:
        writer.write(d)

with jsonlines.open("/aifs4su/rubickjiang/unlearning/data/self_generated_base/qasc/correct_answer.jsonl", "w") as writer:
    for d in ground_data:
        writer.write(d)