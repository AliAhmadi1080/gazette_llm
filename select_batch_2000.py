import os
import json
import pandas as pd

DATA_DIR = "data"
EXCEL_PATH = os.path.join(DATA_DIR, "train-data.xlsx")
PHASE2_OUT = os.path.join(DATA_DIR, "final_corrected_data.json")
PHASE1_OUT = os.path.join(DATA_DIR, "structured_gazette_results.json")
SELECTED_BATCH_OUT = os.path.join(DATA_DIR, "selected_batch_indices.json")

# کلمات کلیدی متمرکز بر ۴ کلاس بحرانی زیر ۷۰٪
TARGETED_KEYWORDS = [
    # ۱. اختیارات مدیرعامل (CEO_AUTHORITY - F1: 26%)
    "اختیارات مدیرعامل", "تفویض شد", "حدود اختیارات", "اساسنامه به مدیرعامل",
    # ۲. موضوع فعالیت (SUBJECT_OF_ACTIVITY - F1: 32%)
    "موضوع فعالیت", "موضوع شرکت", "ماده مربوطه در اساسنامه اصلاح",
    # ۳. قوانین امضا (SIGN_RULE - F1: 48%)
    "حق امضا", "اوراق بهادار", "دارندگان امضا", "مکاتبات اداری با امضا", "مهر شرکت معتبر",
    # ۴. آدرس (ADDRESS - F1: 48%)
    "نشانی شرکت", "تغییر مرکز اصلی", "کد پستی", "تغییر آدرس", "اقامتگاه قانونی"
]

def scan_and_select_targeted_batch():
    # ۱. بارگذاری ردیف‌های پردازش شده قبلی جهت جلوگیری از هم‌پوشانی
    processed_indices = set()
    source_path = PHASE2_OUT if os.path.exists(PHASE2_OUT) else PHASE1_OUT
    
    if os.path.exists(source_path):
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            processed_indices.update(entry["row_index"] for entry in data if "row_index" in entry)
    
    print(f"📊 ردیف‌های از قبل پردازش شده در دیتابیس شما: {len(processed_indices)} نمونه")

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"اکسل در مسیر {EXCEL_PATH} یافت نشد.")
        
    print("🔍 در حال جستجوی فوق‌هدفمند اکسل برای شکار ردیف‌های دارای متغیرهای طولانی بحرانی...")
    df = pd.read_excel(EXCEL_PATH)
    
    targeted_indices = []
    
    for idx, row in df.iterrows():
        if idx in processed_indices:
            continue
            
        text = str(row.get("NewsText", ""))
        if pd.isna(text) or not text.strip():
            continue
            
        # بررسی وجود کلیدواژه‌های بحرانی
        matched_count = sum(1 for kw in TARGETED_KEYWORDS if kw in text)
        
        # اگر حداقل ۲ کلیدواژه بحرانی در متن وجود داشته باشد، آن را انتخاب کن
        if matched_count >= 2:
            targeted_indices.append(int(idx))
            
    print(f"🎯 تعداد {len(targeted_indices)} ردیف فوق‌هدفمند بدون هم‌پوشانی یافت شد.")

    # انتخاب ۲۰۰۰ نمونه آخر بر اساس نظر شما
    selected_batch = targeted_indices[-2000:]
    print(f"📦 انتخاب {len(selected_batch)} نمونه نهایی هدفمند برای فردا.")

    with open(SELECTED_BATCH_OUT, 'w', encoding='utf-8') as f:
        json.dump(selected_batch, f, ensure_ascii=False, indent=2)
    print(f"💾 بچ هدفمند جدید با موفقیت در '{SELECTED_BATCH_OUT}' ذخیره شد.")

if __name__ == "__main__":
    scan_and_select_targeted_batch()