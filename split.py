import json
import random 

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

data = load_jsonl('./data/gsm8k/train.jsonl')

random.seed(42)
random.shuffle(data)

val_ratio = 0.1
n_val = int(len(data) * val_ratio)

val_data = data[:n_val]
train_data = data[n_val:]

save_jsonl("./data/gsm8k/train_split.jsonl", train_data)
save_jsonl("./data/gsm8k/val.jsonl", val_data)

print(len(train_data), len(val_data))