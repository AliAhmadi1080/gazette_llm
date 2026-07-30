import os
import pandas as pd

EXCEL_PATH = os.path.join("data", "train-data.xlsx")
OUTPUT_HIGH_VALUE_PATH = os.path.join("data", "high_value_indices.json")

# واژگان کلیدی تخصصی مربوط به کلاس‌های کمیاب و ضعیف (اقلیت)
RARE_KEYWORDS = [
    # اختیارات مدیرعامل (CEO_AUTHORITY)
    "اختیارات مدیرعامل", "تفویض شد", "حدود اختیارات", "اساسنامه به مدیرعامل",
    # قواعد امضا (SIGN_RULE)
    "حق امضا", "اوراق بهادار", "دارندگان امضا", "مکاتبات اداری با امضا", "مهر شرکت معتبر",
    # سرمایه شرکت (COMPANY_CAPITAL)
    "سرمایه شرکت", "منقسم به", "ریال افزایش", "ریالی بی نام"
]

def scan_excel():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"اکسل در مسیر {EXCEL_PATH} یافت نشد.")
        
    print("📖 در حال اسکن کل ۲۶,۰۰۰ سطر اکسل برای شکار سطر‌های طلایی...")
    df = pd.read_excel(EXCEL_PATH)
    
    high_value_indices = []
    
    for idx, row in df.iterrows():
        text = str(row.get("NewsText", ""))
        if pd.isna(text) or not text.strip():
            continue
            
        # بررسی وجود کلیدواژه‌های طلایی در متن آگهی
        matched_count = sum(1 for kw in RARE_KEYWORDS if kw in text)
        
        # اگر حداقل ۲ کلیدواژه نادر در متن وجود داشته باشد، آن سطر دارای ارزش اطلاعاتی بالا است
        if matched_count >= 2:
            high_value_indices.append(int(idx))
            
    print(f"🎯 تعداد {len(high_value_indices)} سطر فوق‌العاده باارزش و متمرکز پیدا شد!")
    
    with open(OUTPUT_HIGH_VALUE_PATH, 'w', encoding='utf-8') as f:
        json.dump(high_value_indices, f, ensure_ascii=False, indent=2)
    print(f"💾 لیست ایندکس‌ها در '{OUTPUT_HIGH_VALUE_PATH}' ذخیره شد.")

if __name__ == "__main__":
    import json
    scan_excel()