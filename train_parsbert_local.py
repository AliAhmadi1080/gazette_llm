import os
import json
import torch
import numpy as np
import evaluate
from transformers import (
    BertTokenizerFast, 
    BertForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    set_seed  # <--- قفل سید
)
from datasets import load_from_disk

from precision_booster import NERPrecisionBooster

# ۱. قفل کردن تمام متغیرهای تصادفی برای تکرارپذیری دقیق نتایج
set_seed(42)

LABEL_MAP_PATH = os.path.join("data", "label_mapping.json")
DATASET_DIR = "aligned_ner_dataset"

# لود کردن نگاشت لیبل‌ها
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"نقشه لیبل‌ها در مسیر '{LABEL_MAP_PATH}' یافت نشد.")
    
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
    label_to_id = mapping["label_to_id"]
    id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}
    label_list = list(label_to_id.keys())

booster = NERPrecisionBooster(id_to_label=id_to_label, confidence_threshold=0.80)

# ۲. بارگذاری دیتاست آماده‌شده
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"دیتاست آماده‌شده یافت نشد. ابتدا فایل pipeline.py را اجرا کنید.")
full_dataset = load_from_disk(DATASET_DIR)

# تقسیم عادلانه داده‌ها به دو بخش تمرینی (۸۰٪) و ارزیابی محلی (۲۰٪)
ds = full_dataset.train_test_split(test_size=0.2, seed=42)
print(f"📊 وضعیت توزیع داده‌ها: {len(ds['train'])} نمونه آموزش | {len(ds['test'])} نمونه تست محلی.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ فرآیند آموزش روی بستر سخت‌افزاری پردازش خواهد شد: {device.upper()}")

# بازگشت به پارس‌برت بومی و پایدار فارسی
model_name = "./local_model"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

model = BertForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id,
    ignore_mismatched_sizes=True  
).to(device)

metric = evaluate.load("seqeval")

# تابع compute_metrics ساده و بدون بوستر برای داخل فایل آموزش
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2) # آرگ‌مکس ساده و بدون فیلتر آستانه

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# تنظیمات بهینه شده پایدار با استفاده از تکنولوژی BF16 کارت گرافیک 5060 Ti
training_args = TrainingArguments(
    output_dir="./parsbert_finetuned",
    eval_strategy="epoch",            
    save_strategy="epoch",            
    learning_rate=2e-5,               
    per_device_train_batch_size=16,   
    per_device_eval_batch_size=32,    
    num_train_epochs=15,              
    weight_decay=0.05,                
    warmup_ratio=0.1,                 
    logging_steps=10,
    bf16=True,                        # <--- فعال‌سازی BF16 برای پایداری در کارت گرافیک شما
    fp16=False,                       
    load_best_model_at_end=True,      
    metric_for_best_model="f1",       
    report_to="none"                  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🚀 شروع مجدد فاز فاین‌تیونینگ مدل پارس‌برت بومی و پایدار...")
trainer.train()

print("\n📊 ارزیابی نهایی مدل بر روی دیتای تست لوکال:")
evaluation_results = trainer.evaluate()
print(f"F1-Score نهایی مدل بومی: {evaluation_results['eval_f1']:.4f}")
print(f"دقت (Precision): {evaluation_results['eval_precision']:.4f}")
print(f"بازخوانی (Recall): {evaluation_results['eval_recall']:.4f}")

model.save_pretrained("./best_parsbert_model")
tokenizer.save_pretrained("./best_parsbert_model")
print("💾 مدل پارس‌برت بومی با موفقیت در پوشه 'best_parsbert_model' ذخیره شد.")