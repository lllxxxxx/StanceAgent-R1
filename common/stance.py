import re


STANCE_TO_ID = {"支持": 0, "反对": 1, "中立": 2}
ID_TO_STANCE = {0: "支持", 1: "反对", 2: "中立"}


def compute_stance_f1(preds: list, labels: list, num_classes: int = 3) -> dict:
    pred_ids = []
    for text in preds:
        match = re.search(r"<answer>(.*?)</answer>", text)
        stance = match.group(1).strip() if match else text.strip()
        pred_ids.append(STANCE_TO_ID.get(stance, -1))

    per_class_f1 = {}
    total_correct = 0
    valid_count = 0

    for c in range(num_classes):
        tp = sum(1 for p, l in zip(pred_ids, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(pred_ids, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(pred_ids, labels) if p != c and l == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[ID_TO_STANCE[c]] = round(f1, 4)
        total_correct += tp

    for p in pred_ids:
        if p != -1:
            valid_count += 1

    macro_f1 = sum(per_class_f1.values()) / num_classes
    accuracy = total_correct / len(labels) if labels else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "f1_支持": per_class_f1["支持"],
        "f1_反对": per_class_f1["反对"],
        "f1_中立": per_class_f1["中立"],
        "accuracy": round(accuracy, 4),
        "valid_ratio": round(valid_count / len(preds), 4) if preds else 0.0,
    }
