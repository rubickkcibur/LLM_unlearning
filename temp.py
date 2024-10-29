import os
import jsonlines
objs = []
with jsonlines.open("/aifs4su/rubickjiang/unlearning/data/self_generated/qasc/wrong_answer.jsonl", "r") as reader:
    for obj in reader:
        objs.append({
            "user": obj["user"],
            "assistant": obj["assistent"]
        })

with jsonlines.open("/aifs4su/rubickjiang/unlearning/data/self_generated/qasc/wrong_answer.jsonl", "w") as writer:
    for obj in objs:
        writer.write(obj)