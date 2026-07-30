import os
import json
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from tqdm import tqdm

LABEL_MAP_PATH = os.path.join("data", "label_mapping.json")
DATASET_DIR = "aligned_ner_dataset"

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
    id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}

# ۱. بارگذاری داده‌های تست محلی (Holdout Test Set)
full_dataset = load_from_disk(DATASET_DIR)
ds = full_dataset.train_test_split(test_size=0.2, seed=42)
test_data = ds["test"]

# ۲. بارگذاری مدل ۷۸.۸ درصدی نهایی شما
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./best_parsbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

metric = evaluate.load("seqeval")
true_predictions = []
true_labels = []

model.eval()
with torch.no_grad():
    for item in tqdm(test_data, desc="🏃 در حال اجرای ارزیابی تفکیک‌شده"):
        inputs = {
            "input_ids": torch.tensor([item["input_ids"]]).to(device),
            "attention_mask": torch.tensor([item["attention_mask"]]).to(device)
        }
        outputs = model(**inputs)
        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)[0]
        label_seq = item["labels"]
        
        clean_preds = []
        clean_labels = []
        for p_id, l_id in zip(predictions, label_seq):
            if l_id != -100:
                clean_preds.append(id_to_label[p_id])
                clean_labels.append(id_to_label[l_id])
        true_predictions.append(clean_preds)
        true_labels.append(clean_labels)

# ۳. استخراج محاسبات تفکیک‌شده کلاس‌ها از seqeval
results = metric.compute(predictions=true_predictions, references=true_labels)

print("\n📊 کارنامه تفصیلی و کلاس‌به‌کلاس مدل (Class-by-Class Diagnostics):")
print("-" * 90)
print(f"{'Class Name':<45} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 90)

weak_classes = []
for key, val in results.items():
    # نادیده گرفتن متغیرهای تجمعی
    if key in ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"]:
        continue
    f1 = val["f1"]
    precision = val["precision"]
    recall = val["recall"]
    
    print(f"{key:<45} | {precision:.4f}     | {recall:.4f}  | {f1:.4f}")
    
    # شناسایی کلاس‌های بحرانی با اف‌وان زیر ۷۰٪
    if f1 < 0.70:
        weak_classes.append(key)

print("-" * 90)
print(f"⚠️ کلاس‌های بحرانی و نیازمند تقویت (F1 < 70%): {weak_classes}")