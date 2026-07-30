# train_relation_classifier.py (آموزش مدل دوم: دسته‌بندی روابط)

import os
import json
import torch
import numpy as np
import evaluate
from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset

# ۱. قفل کردن تصادفی‌سازی
set_seed(42)

DATASET_PATH = os.path.join("data", "relation_classification_dataset.json")
SPECIAL_TOKENS_PATH = os.path.join("data", "relation_special_tokens.json")
MODEL_OUTPUT_DIR = "./best_relation_model"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"فایل دیتابیس روابط در '{DATASET_PATH}' یافت نشد. ابتدا build_relation_dataset.py را اجرا کنید.")

# لیست ۲۶ رابطه مجاز + ۱ کلاس بدون رابطه (O)
RELATION_LABELS = [
    "O",
    "PERSON_TO_POSITION", "PERSON_TO_PERSONAL_ID", "PERSON_TO_COMPANY",
    "PERSON_TO_DURATION_OF_ACTIVITY", "PERSON_TO_PERSON", "PERSON_TO_CORPORATE_ID",
    "POSITION_TO_DURATION_OF_ACTIVITY", "POSITION_TO_POSITION", "POSITION_TO_COMPANY",
    "POSITION_TO_CEO_AUTHORITY", "COMPANY_TO_CORPORATE_ID", "COMPANY_TO_POSITION",
    "COMPANY_TO_CORPORATE_REGISTRATION_NUMBER", "COMPANY_TO_DATE", "COMPANY_TO_ADDRESS",
    "COMPANY_TO_SUBJECT_OF_ACTIVITY", "COMPANY_TO_DURATION_OF_ACTIVITY",
    "COMPANY_TO_PERSONAL_ID", "COMPANY_TO_PERSON", "COMPANY_TO_COMPANY",
    "COMPANY_TO_COMPANY_CAPITAL", "COMPANY_TO_CEO_AUTHORITY",
    "SUBJECT_OF_ACTIVITY_TO_DATE", "SUBJECT_OF_ACTIVITY_TO_DURATION_OF_ACTIVITY",
    "DURATION_OF_ACTIVITY_TO_DATE", "DURATION_OF_ACTIVITY_TO_POSITION"
]

label2id = {l: i for i, l in enumerate(RELATION_LABELS)}
id2label = {i: l for i, l in enumerate(RELATION_LABELS)}

# ۲. بارگذاری توکنایزر و افزودن توکن‌های ویژه نشانه‌گذار
model_name = "./local_model"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

if os.path.exists(SPECIAL_TOKENS_PATH):
    with open(SPECIAL_TOKENS_PATH, 'r', encoding='utf-8') as f:
        special_tokens = json.load(f)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# ۳. بارگذاری و آماده‌سازی داده‌ها
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

processed_samples = []
for item in raw_data:
    lbl = item.get("label", "O")
    if lbl not in label2id:
        lbl = "O"
    processed_samples.append({
        "sentence": item["sentence"],
        "label": label2id[lbl]
    })

hf_dataset = Dataset.from_list(processed_samples)
ds = hf_dataset.train_test_split(test_size=0.2, seed=42)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=512)

tokenized_ds = ds.map(tokenize_function, batched=True)

# ۴. لود کردن مدل دسته‌بندی توالی
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ آموزش مدل دوم دسته‌بندی روابط روی {device.upper()} اجرا می‌شود...")

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(RELATION_LABELS),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

# بازتنظیم لایه امبدینگ برای شناسایی توکن‌های ویژه جدید
model.resize_token_embeddings(len(tokenizer))

# ۵. متد ارزیابی
metric = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    results = metric.compute(predictions=preds, references=p.label_ids, average="macro")
    return {"macro_f1": results["f1"]}

# ۶. تنظیمات بهینه آموزش با BF16 برای کارت گرافیک 5060 Ti
training_args = TrainingArguments(
    output_dir="./relation_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32, # بچ‌سایز بزرگ برای سرعت بالا
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=20,
    bf16=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🚀 شروع فاین‌تیونینگ مدل دوم (Relation Classifier)...")
trainer.train()

print("\n📊 ارزیابی نهایی مدل دوم روی دیتای تست لوکال:")
eval_res = trainer.evaluate()
print(f"Macro F1 مدل دوم: {eval_res['eval_macro_f1']:.4f}")

model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"💾 مدل دوم با موفقیت در پوشه '{MODEL_OUTPUT_DIR}' ذخیره شد.")