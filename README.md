# 🏛️ AI-Newspaper: Official Gazette Entity & Relation Extraction
### IranAI 1404 - Research Institute for ICT Challenge

این مخزن شامل کدبیس کامل، پایپ‌لاین‌های داده، اسکریپت‌های آموزش مدل‌های عمیق و موتورهای استخراج پروفایل برای مسابقه «استخراج موجودیت‌ها و روابط حقوقی از متون روزنامه رسمی» است.

---

## 📊 نتایج ارزیابی نهایی (Leaderboard Benchmarks)

* **روش اول (موتور ترکیبی الگو و لنگرکشی):** **Macro Score: 70.93%** | Micro Precision: 84.54%
  * `PERSON`: **F1 = 90.14%** (Precision = 100%)
  * `PERSONAL_ID`: **F1 = 96.27%** (Precision = 100%)
* **روش دوم (پایپ‌لاین عمیق ParsBERT + Relation Classifier):** **Macro Score: 58.42%** | Micro Precision: 76.29%

---

## 📁 ساختار دایرکتوری‌ها و فایل‌های پروژه

```text
AI-NEWSPAPER/
├── data/                               # پوشه ذخیره‌سازی داده‌های خام و پردازش‌شده
│   ├── train-data.xlsx                 # فایل اکسل داده‌های آموزشی
│   ├── train.xlsx / validation.xlsx    # اسناد اصلی مسابقه
│   ├── validation_people_names.csv     # لیست ۱۵۳ فرد هدف
│   ├── final_corrected_data.json       # ۵,۵۲۰ دادهٔ ممیزی‌شده
│   └── label_mapping.json              # نگاشت لیبل‌های ۱۳‌گانه
├── aligned_ner_dataset/                # دیتاست هم‌تراز‌شده توکن‌ها برای ParsBERT
├── best_parsbert_model/                # بهترین وزن‌های ذخیره‌شده مدل NER (ParsBERT)
├── best_relation_model/                # بهترین وزن‌های ذخیره‌شده مدل دوم (Relation Classifier)
├── local_model/                        # وزن‌های مدل پایه
├── active_learning_scanner.py          # اسکنر Active Learning برای شکار کلاس‌های کمیاب
├── build_relation_dataset.py           # ساخت دیتابیس روابط با توکن‌های نشانه‌گذار [S:TYPE]
├── calibrate_threshold.py              # کالیبراسیون آستانه اطمینان Softmax
├── class_diagnostics.py                # اسکریپت آنالیز جراحی و خطای کلاس‌به‌کلاس
├── clean_final_submission.py           # جلاکاری و پاکسازی کدهای ملی سرایت‌کرده
├── gemini_pipeline.py                  # پایپ‌لاین استخراج ابری موازی با Google Gemini API
├── pipeline_inference_phase3.py        # استنباط یکپارچه دو مدله (NER + RE)
├── precision_booster.py                # ماژول تقویت دقت و اصلاح گرامر توکن‌ها
├── solve_phase3_excel.py               # موتور استخراج لنگری محلی (ثبت امتیاز ۷۰.۹۳٪ لیدربورد)
├── train_parsbert_local.py             # آموزش مدل اول (ParsBERT NER) با BF16 و Seed Lock
└── train_relation_classifier.py        # آموزش مدل دوم (Relation Classifier)
```

---

## 🚀 نحوه اجرای پروژه

### ۱. پیش‌نیازها و نصب کتابخانه‌ها
```bash
pip install torch transformers datasets pandas openpyxl evaluate seqeval tqdm
```

### ۲. آموزش مدل اول (ParsBERT NER)
```bash
python train_parsbert_local.py
```

### ۳. ساخت دیتابیس و آموزش مدل دوم (Relation Classifier)
```bash
python build_relation_dataset.py
python train_relation_classifier.py
```

### ۴. ساخت فایل سابمیشن نهایی
```bash
python solve_phase3_excel.py
```
*خروجی نهایی در فایل `submission_phase3_excel.json` ذخیره می‌شود.*

---
**توسعه‌یافته برای مسابقه هوش مصنوعی روزنامه رسمی (۱۴۰۵ / ۲۰۲۶)**
```

---

تمام مستندات، ساختار فایل Word، گزارش علمی ۷ بخشی و فایل README.md با رعایت تمامی استانداردها و قیود اعلام‌شده توسط شما آماده گردید.
