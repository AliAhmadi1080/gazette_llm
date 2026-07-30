# solve_phase3_ultimate.py (نسخه نهایی با فیلتر الگوریتم الگوی باقی‌مانده کد ملی و استخراج کامل شرکت‌ها)

import os
import re
import json
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data"
TRAIN_EXCEL = os.path.join(DATA_DIR, "train.xlsx")
VAL_EXCEL = os.path.join(DATA_DIR, "validation.xlsx")
VAL_NAMES_CSV = os.path.join(DATA_DIR, "validation_people_names.csv")
OUTPUT_JSON_PATH = "submission_phase3_gold.json"

VALID_ATTRIBUTE_KEYS = [
    "POSITION", "CEO_AUTHORITY", "COMPANY", "CORPORATE_ID",
    "CORPORATE_REGISTRATION_NUMBER", "COMPANY_CAPITAL", "ADDRESS",
    "DATE", "DURATION_OF_ACTIVITY", "SIGN_RULE", "SUBJECT_OF_ACTIVITY"
]

def is_valid_iranian_national_id(code: str) -> bool:
    """صحت‌سنجی الگوریتمی کد ملی ایران جهت حذف کدهای پیگیری آگهی (98...)"""
    if not re.match(r'^\d{10}$', code):
        return False
    # کدهای پیگیری آگهی یا شناسه شرکت‌ها
    if code.startswith("98") or code.startswith("139") or code.startswith("10") or code.startswith("14"):
        return False
    
    # محاسبه الگوریتم رسمی باقی‌مانده کد ملی
    check = int(code[9])
    s = sum(int(code[i]) * (10 - i) for i in range(9)) % 11
    return (s < 2 and check == s) or (s >= 2 and check == 11 - s)

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'#x0[DdAa];', ' ', text)
    text = text.replace('\u0643', '\u06a9').replace('\u064a', '\u06cc')
    return " ".join(text.split())

def extract_strict_profile_for_person(text, target_name):
    pids = set()
    positions = set()
    companies = set()
    corporate_ids = set()
    dates = set()

    # ۱. استخراج کد ملی با لنگراندازی دقیق به نام فرد و اعتبار سنجی الگوریتمی
    pid_patterns = [
        rf"{re.escape(target_name)}[^\d\n\.]*?(?:کدملی|کد ملی|شماره ملی|ش م)[^\d]*?(\d{{10}})",
        rf"(?:آقای|خانم)\s+{re.escape(target_name)}[^\d\n\.]*?(\d{{10}})"
    ]
    for pattern in pid_patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if is_valid_iranian_national_id(m):
                pids.add(m)

    # ۲. استخراج سمت و شرکت در مجاورت جمله مستقیم فرد
    positions_list = [
        "نایب رئیس هیئت مدیره", "نائب رئیس هیئت مدیره", "رئیس هیئت مدیره", 
        "رئیس هیات مدیره", "مدیرعامل", "عضو هیئت مدیره", "عضو هیات مدیره",
        "عضوهیئت مدیره", "عضوهیات مدیره", "بازرس اصلی", "بازرس علی البدل", "منشی هیئت مدیره"
    ]
    
    sentences = re.split(r'[\n\.]| و | - | ـ ', text)
    for sent in sentences:
        if target_name in sent:
            # الف) سمت دقیق فرد
            for pos in positions_list:
                if pos in sent:
                    positions.add(pos)
                    
            # ب) استخراج کامل نام شرکت (جلوگیری از قطع شدن نام شرکت)
            comp_match = re.search(r'نمایندگی\s+از\s+((?:شرکت|موسسه|سازمان)\s+[^،\.\n]+?)(?=\s+(?:به|شماره|شناسه|ثبت|به سمت|تعیین|انتخاب|$))', sent)
            if comp_match:
                full_comp = comp_match.group(1).strip()
                if len(full_comp) > 3:
                    companies.add(full_comp)
                    # استخراج شناسه ملی همان شرکت
                    c_id_match = re.search(rf"{re.escape(full_comp)}[^\d]*?(?:شناسه ملی)\s*(\d{{11}})", text)
                    if c_id_match:
                        corporate_ids.add(c_id_match.group(1))

    # اگر فرد در این سند نقشی داشته باشد، تاریخ آگهی ثبت می‌شود
    if pids or positions or companies:
        d_matches = re.findall(r'\b\d{2,4}/\d{1,2}/\d{2,4}\b', text)
        dates.update(d_matches)

    return pids, positions, companies, corporate_ids, dates

def run_ultimate_pipeline():
    print("📖 در حال اجرای اسکریپت طلایی و پالایش‌شده فاز ۳...")
    df_val_names = pd.read_csv(VAL_NAMES_CSV)
    df_train = pd.read_excel(TRAIN_EXCEL) if os.path.exists(TRAIN_EXCEL) else pd.DataFrame()
    df_val = pd.read_excel(VAL_EXCEL) if os.path.exists(VAL_EXCEL) else pd.DataFrame()
    
    all_docs = []
    if not df_train.empty: all_docs.append(df_train)
    if not df_val.empty: all_docs.append(df_val)
    
    df_corpus = pd.concat(all_docs, ignore_index=True)
    df_corpus['clean_text'] = df_corpus['NewsText'].apply(clean_text)
    
    final_output = []
    
    for _, q_row in tqdm(df_val_names.iterrows(), total=len(df_val_names)):
        q_id = int(q_row['query_id'])
        target_name = str(q_row['name']).strip()
        
        matched_names = set([target_name])
        matched_pids = set()
        matched_attributes = {k: set() for k in VALID_ATTRIBUTE_KEYS}
        
        for _, doc_row in df_corpus.iterrows():
            text = doc_row['clean_text']
            if target_name not in text:
                continue
                
            pids, pos_set, comp_set, corp_ids, dates = extract_strict_profile_for_person(text, target_name)
            
            matched_pids.update(pids)
            matched_attributes["POSITION"].update(pos_set)
            matched_attributes["COMPANY"].update(comp_set)
            matched_attributes["CORPORATE_ID"].update(corp_ids)
            matched_attributes["DATE"].update(dates)

        clean_attrs = {}
        for k, v in matched_attributes.items():
            if v:
                clean_attrs[k] = list(v)
                
        final_output.append({
            "query_id": q_id,
            "profile": {
                "names": list(matched_names),
                "personal_ids": list(matched_pids),
                "attributes": clean_attrs
            }
        })
        
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 فایل طلایی و بدون باگ در '{OUTPUT_JSON_PATH}' ذخیره شد.")

if __name__ == "__main__":
    run_ultimate_pipeline()