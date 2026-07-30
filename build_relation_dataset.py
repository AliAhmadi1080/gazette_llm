# build_relation_dataset.py (ساخت دیتابیس هوشمند برای مدل دسته‌بندی روابط)

import os
import re
import json
import random
import pandas as pd
from tqdm import tqdm

# مسیرهای داده‌ها منطبق بر ساختار پروژه
DATA_DIR = "data"
EXCEL_PATH = os.path.join(DATA_DIR, "train-data.xlsx")
PHASE1_OUT = os.path.join(DATA_DIR, "structured_gazette_results.json")
PHASE2_OUT = os.path.join(DATA_DIR, "final_corrected_data.json")

# خروجی‌های اصلی این اسکریپت
RELATION_DATASET_OUT = os.path.join(DATA_DIR, "relation_classification_dataset.json")
RELATION_SPECIAL_TOKENS_OUT = os.path.join(DATA_DIR, "relation_special_tokens.json")

# ۱۳ نوع موجودیت مجاز مسابقه
VALID_ENTITY_TYPES = [
    "PERSON", "PERSONAL_ID", "POSITION", "CEO_AUTHORITY",
    "COMPANY", "CORPORATE_ID", "CORPORATE_REGISTRATION_NUMBER",
    "COMPANY_CAPITAL", "ADDRESS", "DATE", "DURATION_OF_ACTIVITY",
    "SIGN_RULE", "SUBJECT_OF_ACTIVITY"
]

# ماتریس ساختاری جفت‌های مبدا و مقصد مجاز برای روابط
VALID_RELATION_MAPS = {
    "PERSON_TO_POSITION": ("PERSON", "POSITION"),
    "PERSON_TO_PERSONAL_ID": ("PERSON", "PERSONAL_ID"),
    "PERSON_TO_COMPANY": ("PERSON", "COMPANY"),
    "PERSON_TO_DURATION_OF_ACTIVITY": ("PERSON", "DURATION_OF_ACTIVITY"),
    "PERSON_TO_PERSON": ("PERSON", "PERSON"),
    "PERSON_TO_CORPORATE_ID": ("PERSON", "CORPORATE_ID"),
    "POSITION_TO_DURATION_OF_ACTIVITY": ("POSITION", "DURATION_OF_ACTIVITY"),
    "POSITION_TO_POSITION": ("POSITION", "POSITION"),
    "POSITION_TO_COMPANY": ("POSITION", "COMPANY"),
    "POSITION_TO_CEO_AUTHORITY": ("POSITION", "CEO_AUTHORITY"),
    "COMPANY_TO_CORPORATE_ID": ("COMPANY", "CORPORATE_ID"),
    "COMPANY_TO_POSITION": ("COMPANY", "POSITION"),
    "COMPANY_TO_CORPORATE_REGISTRATION_NUMBER": ("COMPANY", "CORPORATE_REGISTRATION_NUMBER"),
    "COMPANY_TO_DATE": ("COMPANY", "DATE"),
    "COMPANY_TO_ADDRESS": ("COMPANY", "ADDRESS"),
    "COMPANY_TO_SUBJECT_OF_ACTIVITY": ("COMPANY", "SUBJECT_OF_ACTIVITY"),
    "COMPANY_TO_DURATION_OF_ACTIVITY": ("COMPANY", "DURATION_OF_ACTIVITY"),
    "COMPANY_TO_PERSONAL_ID": ("COMPANY", "PERSONAL_ID"),
    "COMPANY_TO_PERSON": ("COMPANY", "PERSON"),
    "COMPANY_TO_COMPANY": ("COMPANY", "COMPANY"),
    "COMPANY_TO_COMPANY_CAPITAL": ("COMPANY", "COMPANY_CAPITAL"),
    "COMPANY_TO_CEO_AUTHORITY": ("COMPANY", "CEO_AUTHORITY"),
    "SUBJECT_OF_ACTIVITY_TO_DATE": ("SUBJECT_OF_ACTIVITY", "DATE"),
    "SUBJECT_OF_ACTIVITY_TO_DURATION_OF_ACTIVITY": ("SUBJECT_OF_ACTIVITY", "DURATION_OF_ACTIVITY"),
    "DURATION_OF_ACTIVITY_TO_DATE": ("DURATION_OF_ACTIVITY", "DATE"),
    "DURATION_OF_ACTIVITY_TO_POSITION": ("DURATION_OF_ACTIVITY", "POSITION")
}


def clean_legal_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'#x0[DdAa];', ' ', text)
    bidi_chars_pattern = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')
    text = bidi_chars_pattern.sub('', text)
    text = text.replace('\u0643', '\u06a9').replace('\u064a', '\u06cc')
    return " ".join(text.split())


def generate_special_tokens():
    """تولید خودکار ۵۲ توکن نشانه‌گذار تایپ‌بندی‌شده بر اساس ۱۳ نوع موجودیت"""
    special_tokens = []
    for ent_type in VALID_ENTITY_TYPES:
        special_tokens.extend([
            f"[S:{ent_type}]", f"[/S:{ent_type}]",
            f"[O:{ent_type}]", f"[/O:{ent_type}]"
        ])
    return special_tokens


def insert_entity_markers(text, subj_text, subj_type, obj_text, obj_type):
    """جایگذاری امین تگ‌های نشانه‌گذار بدون بهم‌ریختگی طول رشته (از انتها به ابتدا)"""
    subj_matches = list(re.finditer(re.escape(subj_text), text))
    obj_matches = list(re.finditer(re.escape(obj_text), text))

    if not subj_matches or not obj_matches:
        return None

    # انتخاب اولین رخداد هر موجودیت در متن
    subj_start, subj_end = subj_matches[0].span()
    obj_start, obj_end = obj_matches[0].span()

    # مدیریت تداخل فیزیکی موجودیت‌ها در متن
    if (subj_start <= obj_start < subj_end) or (obj_start <= subj_start < obj_end):
        return None

    s_open, s_close = f"[S:{subj_type}]", f"[/S:{subj_type}]"
    o_open, o_close = f"[O:{obj_type}]", f"[/O:{obj_type}]"

    # تگ‌گذاری بر اساس موقعیت مکانی جهت حفظ ایندکس‌ها
    if subj_start < obj_start:
        tagged_text = (
            text[:obj_start] + o_open + obj_text + o_close + text[obj_end:]
        )
        tagged_text = (
            tagged_text[:subj_start] + s_open + subj_text + s_close + tagged_text[subj_end:]
        )
    else:
        tagged_text = (
            text[:subj_start] + s_open + subj_text + s_close + text[subj_end:]
        )
        tagged_text = (
            tagged_text[:obj_start] + o_open + obj_text + o_close + tagged_text[obj_end:]
        )

    return tagged_text


def build_dataset():
    source_path = PHASE2_OUT if os.path.exists(PHASE2_OUT) else PHASE1_OUT
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"هیچ دیتای استخراج‌شده‌ای در مسیر '{source_path}' یافت نشد. ابتدا پایپ‌لاین را اجرا کنید.")
    
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"فایل اکسل در مسیر '{EXCEL_PATH}' یافت نشد.")

    print(f"📖 در حال لود کردن داده‌ها از اکسل و منبع استخراج ابری '{source_path}'...")
    df_excel = pd.read_excel(EXCEL_PATH)
    df_excel['cleaned_text'] = df_excel['NewsText'].apply(clean_legal_text)

    with open(source_path, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)

    # ایجاد و ذخیره فایل توکن‌های ویژه جهت اضافه شدن به توکنایزر
    special_tokens = generate_special_tokens()
    with open(RELATION_SPECIAL_TOKENS_OUT, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    print(f"💾 {len(special_tokens)} توکن ویژه در فایل '{RELATION_SPECIAL_TOKENS_OUT}' ثبت شد.")

    relation_dataset = []
    positive_count = 0
    negative_count = 0

    print("⚡ شروع فرآیند نشانه‌گذاری هوشمند روی متون...")
    for entry in tqdm(extracted_data):
        row_idx = entry.get("row_index")
        if row_idx is None or row_idx >= len(df_excel):
            continue

        original_text = df_excel['cleaned_text'].iloc[row_idx]
        if not original_text:
            continue

        entities = entry.get("entities", [])
        relations = entry.get("relations", [])

        entity_map = {clean_legal_text(ent["text"]): ent["type"] for ent in entities if ent.get("text")}

        # ۱. ساخت نمونه‌های مثبت (روابط واقعی تایید شده)
        positive_keys = set()
        for rel in relations:
            from_ent = clean_legal_text(rel.get("from_entity"))
            to_ent = clean_legal_text(rel.get("to_entity"))
            rel_type = rel.get("relation_type")

            if not from_ent or not to_ent or not rel_type:
                continue

            from_type = entity_map.get(from_ent)
            to_type = entity_map.get(to_ent)

            if not from_type or not to_type:
                continue

            tagged_sentence = insert_entity_markers(original_text, from_ent, from_type, to_ent, to_type)
            if tagged_sentence:
                relation_dataset.append({
                    "row_index": row_idx,
                    "sentence": tagged_sentence,
                    "label": rel_type,
                    "subj": from_ent,
                    "obj": to_ent
                })
                positive_count += 1
                positive_keys.add((from_ent, to_ent))

        # ۲. استخراج هوشمند نمونه‌های منفی (Hard Negatives)
        local_negative_candidates = []
        entity_list = list(entity_map.keys())

        for i in range(len(entity_list)):
            for j in range(len(entity_list)):
                if i == j: continue

                ent_A = entity_list[i]
                ent_B = entity_list[j]

                if (ent_A, ent_B) in positive_keys: continue

                type_A = entity_map[ent_A]
                type_B = entity_map[ent_B]

                # بررسی اینکه آیا ساختار تایپ این دو موجودیت قابلیت داشتن رابطه دارد یا خیر
                is_valid_signature = False
                for sig_from, sig_to in VALID_RELATION_MAPS.values():
                    if type_A == sig_from and type_B == sig_to:
                        is_valid_signature = True
                        break

                if is_valid_signature:
                    tagged_sentence = insert_entity_markers(original_text, ent_A, type_A, ent_B, type_B)
                    if tagged_sentence:
                        local_negative_candidates.append({
                            "row_index": row_idx,
                            "sentence": tagged_sentence,
                            "label": "O", # کلاس بدون رابطه
                            "subj": ent_A,
                            "obj": ent_B
                        })

        # توازن پویا: حداکثر ۲ برابر روابط مثبت، رابطه منفی نمونه‌برداری کن
        max_negatives_allowed = max(len(positive_keys) * 2, 2)
        if len(local_negative_candidates) > max_negatives_allowed:
            local_negative_candidates = random.sample(local_negative_candidates, max_negatives_allowed)

        relation_dataset.extend(local_negative_candidates)
        negative_count += len(local_negative_candidates)

    # ذخیره نهایی
    with open(RELATION_DATASET_OUT, 'w', encoding='utf-8') as f:
        json.dump(relation_dataset, f, ensure_ascii=False, indent=2)

    print("\n📊 آمار دیتابیس ساخته‌شده برای مدل دوم (Relation Classifier):")
    print(f"✅ تعداد کل نمونه‌ها: {len(relation_dataset)}")
    print(f"➕ نمونه‌های مثبت (روابط واقعی): {positive_count}")
    print(f"➖ نمونه‌های منفی (عدم رابطه): {negative_count}")
    print(f"💾 دیتابیس نهایی با موفقیت در '{RELATION_DATASET_OUT}' ذخیره شد.")

if __name__ == "__main__":
    build_dataset()