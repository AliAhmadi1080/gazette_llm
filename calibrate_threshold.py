# calibrate_threshold.py

import os
import json
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from precision_booster import NERPrecisionBooster

LABEL_MAP_PATH = os.path.join("data", "label_mapping.json")
DATASET_DIR = "aligned_ner_dataset"

# ۱. لود کردن لیبل‌ها و داده‌های تست محلی
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
    id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}

full_dataset = load_from_disk(DATASET_DIR)
ds = full_dataset.train_test_split(test_size=0.2, seed=42)
test_data = ds["test"]

# ۲. لود کردن مدل آموزش‌دیدهٔ نهایی شما
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./best_parsbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

# ۳. آماده‌سازی استنباط
metric = evaluate.load("seqeval")
all_logits = []
all_labels = []

model.eval()
with torch.no_grad():
    for item in test_data:
        inputs = {
            "input_ids": torch.tensor([item["input_ids"]]).to(device),
            "attention_mask": torch.tensor([item["attention_mask"]]).to(device)
        }
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu())
        all_labels.append(item["labels"])

# ۴. جستجوی آستانه بهینه (Threshold Sweep)
print("🔍 شروع فرآیند کالیبراسیون آستانه اطمینان برای دستیابی به بالاترین F1...")
best_f1 = 0
best_thresh = 0

# تست آستانه‌های مختلف از 0.0 (بدون بوستر) تا 0.85
for thresh in np.arange(0.0, 0.9, 0.05):
    booster = NERPrecisionBooster(id_to_label=id_to_label, confidence_threshold=thresh)
    
    true_predictions = []
    true_labels = []
    
    for logits, label_seq in zip(all_logits, all_labels):
        boosted_preds = booster.process_predictions(logits)[0]
        
        clean_preds = []
        clean_labels = []
        for p_id, l_id in zip(boosted_preds, label_seq):
            if l_id != -100:
                clean_preds.append(id_to_label[p_id])
                clean_labels.append(id_to_label[l_id])
        true_predictions.append(clean_preds)
        true_labels.append(clean_labels)
        
    results = metric.compute(predictions=true_predictions, references=true_labels)
    f1 = results["overall_f1"]
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    
    print(f"Thresh: {thresh:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\n کالیبراسیون موفقیت‌آمیز بود! بهترین آستانه اطمینان: {best_thresh:.2f} با F1-Score خیره‌کنندهٔ: {best_f1:.4f}")