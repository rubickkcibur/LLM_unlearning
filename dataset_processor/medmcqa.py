from datasets import load_dataset
import re
COT_EXAMPLES_chat = [
    dict(role="user", content="Question: Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma?\n(A) Hyperplasia\n(B) Hyperophy\n(C) Atrophy\n(D) Dyplasia\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine. The answer is (C).\n"),
    dict(role="user", content="Question: Which vitamin is supplied from only animal source:?\n(A) Vitamin C\n(B) Vitamin B7\n(C) Vitamin B12\n(D) Vitamin D\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: Vitamin B12 (Cobalamin) is synthesized solely by microorganisms.* In humans, the only source for humans is food of animal origin, e.g., meat, fish, and dairy products.* Vegetables, fruits, and other foods of nonanimal origin doesn't contain Vitamin B12 .* Daily requirements of vitamin Bp is about 1-3 pg. Body stores are of the order of 2-3 mg, sufficient for 3-4 years if supplies are completely cut off. The answer is (C).\n"),
    dict(role="user", content="Question: Growth hormone has its effect on growth through?\n(A) Directly\n(B) IG1-1\n(C) Thyroxine\n(D) Intranuclear receptors\nLet's think step by step.\n"),
    dict(role="assistant", content="Answer: GH has two major functions :-i) Growth of skeletal system :- The growth is mediated by somatomedins (IGF). Increased deposition of cailage (including chondroitin sulfate) and bone with increased proliferation of chondrocytes and osteocytes.ii) Metabolic effects :- Most of the metabolic effects are due to direct action of GH. These include gluconeogenesis, decreased peripheral utilization of glucose (decreased uptake), lipolysis and anabolic effect on proteins. The answer is (B).\n"),
]

COT_EXAMPLES_base = [
    d["content"]
    for d in COT_EXAMPLES_chat
]
COT_EXAMPLES_base = "".join(COT_EXAMPLES_base)

def train_data():
    option_map = ["A", "B", "C", "D"]
    train_data = []
    data = load_dataset("/aifs4su/rubickjiang/public_data/medmcqa", split="train[5:]")
    for d in data:
        train_data.append(
            [
                {"role": "user", "content": "Question: {}?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\nLet's think step by step:\n".format(
                    d["question"],
                    d["opa"],
                    d["opb"],
                    d["opc"],
                    d["opd"],
                )},
                {"role": "assistant", "content": "Answer: {}. The answer is ({}).\n".format(
                    d["exp"],
                    option_map[int(d["cop"])]
                )}
            ]
        )
    return train_data

def self_data():
    option_map = ["A", "B", "C", "D"]
    data = load_dataset("/aifs4su/rubickjiang/public_data/medmcqa", split="train[5:10000]")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content="Question: {}?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\nLet's think step by step:\n".format(
            d["question"],
            d["opa"],
            d["opb"],
            d["opc"],
            d["opd"],
        ))]
        for d in data
    ]
    questions = [
        "Question: {}?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\nLet's think step by step:\n".format(
            d["question"],
            d["opa"],
            d["opb"],
            d["opc"],
            d["opd"],
        )
        for d in data
    ]
    ground = [
        option_map[int(d["cop"])]
        for d in data
    ]
    return {
        "inputs": inputs,
        "questions": questions,
        "ground": ground
    }

def test_data():
    option_map = ["A", "B", "C", "D"]
    data = load_dataset("/aifs4su/rubickjiang/public_data/medmcqa", split="validation")
    inputs = [
        COT_EXAMPLES_chat +
        [dict(role="user", content = "Question: {}?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\nLet's think step by step:\n".format(
                    d["question"],
                    d["opa"],
                    d["opb"],
                    d["opc"],
                    d["opd"],
                ))]
        for d in data
    ]
    ground = [
        option_map[int(d["cop"])]
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
        pattern = r"\(([a-d])\)"
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
