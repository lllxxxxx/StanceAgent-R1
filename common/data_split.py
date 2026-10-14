import random


def split_train(data, ratio, seed):
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    k = int(len(data) * ratio)
    sft_part = [data[i] for i in sorted(indices[:k])]
    grpo_part = [data[i] for i in sorted(indices[k:])]
    return sft_part, grpo_part
