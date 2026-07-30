# gemini_pipeline.py (پایپ‌لاین موازی بر پایه API رسمی Google Gemini)

import os
import re
import json
import queue
import threading
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types

# ==========================================
# پیکربندی سیستم کنترل دسته‌ای ابری موازی (Google Gemini)
# ==========================================
RUN_PHASE_1 = True         # فعال‌سازی فاز ۱ (استخراج اولیه)
RUN_PHASE_2 = True         # فعال‌سازی فاز ۲ (ممیزی و تصحیح)
RUN_PHASE_3 = True         # فعال‌سازی فاز ۳ (هم‌ترازی محلی توکن‌ها برای پارس‌برت)

PHASE_1_BATCH_LIMIT = 5000  # تعداد سطر جدید جهت استخراج در این اجرا

# تعیین تعداد ریسه‌ها (تنظیم ۲ به ۲ بر اساس درخواست شما)
NUM_PHASE_1_THREADS = 7
NUM_PHASE_2_THREADS = 7

# تخصیص کلیدهای API اختصاصی Gemini شما (چرخشی بین ریسه‌ها)
GEMINI_API_KEYS = [
            '' # api keys
]

# مسیرهای دایرکتوری داده‌ها
DATA_DIR = "data"
EXCEL_PATH = os.path.join(DATA_DIR, "train-data.xlsx")
SELECTED_BATCH_PATH = os.path.join(DATA_DIR, "selected_batch_indices.json")
PHASE1_OUT = os.path.join(DATA_DIR, "structured_gazette_results.json")
PHASE2_OUT = os.path.join(DATA_DIR, "final_corrected_data.json")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_mapping.json")
OUTPUT_DATASET_DIR = "aligned_ner_dataset"

os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "gemini-3.1-flash-lite"  # یا "gemini-1.5-flash" بر اساس کلید شما

# ==========================================
# تعاریف ساختاری و قفل‌های ریسه‌ای (Thread Safety Locks)
# ==========================================
phase1_lock = threading.Lock()
phase2_lock = threading.Lock()

phase1_shared_results = []
phase2_shared_results = []

phase1_queue = queue.Queue()
phase2_queue = queue.Queue()

VALID_ENTITY_TYPES = [
    "PERSON", "PERSONAL_ID", "POSITION", "CEO_AUTHORITY",
    "COMPANY", "CORPORATE_ID", "CORPORATE_REGISTRATION_NUMBER",
    "COMPANY_CAPITAL", "ADDRESS", "DATE", "DURATION_OF_ACTIVITY",
    "SIGN_RULE", "SUBJECT_OF_ACTIVITY"
]

PHASE1_USER_INSTRUCTION = """You must extract legal entities and relations from the text and represent them as a single JSON object.

Strict Rules for Extraction and Graph Consistency:
1. ADDRESS UNIFICATION: Never split an address into multiple entities. An address must be extracted as a single, long, unified string in ONE single ADDRESS entity.
2. CORRECT POSTAL CODE CLASSIFICATION: Never classify a 10-digit postal code as PERSONAL_ID or DATE. Postal codes must be kept inside the ADDRESS text string.
3. SIGN_RULE ISOLATION: The SIGN_RULE entity is strictly isolated. It must NEVER have any relations to other entities.
4. ESTABLISH LOGICAL RELATIONS: If you extract related entities, you MUST establish their logical relations.
5. CORPORATE_ID CLASSIFICATION: Any 11-digit number starting with 10 or 14 MUST be classified strictly as `CORPORATE_ID`.
6. JSON KEYS: In "entities" array use only "text" and "type". In "relations" array use only "from_entity", "to_entity" and "relation_type".

Allowed Entity Types:
PERSON, PERSONAL_ID, POSITION, CEO_AUTHORITY, COMPANY, CORPORATE_ID, CORPORATE_REGISTRATION_NUMBER, COMPANY_CAPITAL, ADDRESS, DATE, DURATION_OF_ACTIVITY, SIGN_RULE, SUBJECT_OF_ACTIVITY.

Allowed Relation Types:
PERSON_TO_POSITION, PERSON_TO_PERSONAL_ID, PERSON_TO_COMPANY, PERSON_TO_DURATION_OF_ACTIVITY, PERSON_TO_PERSON, PERSON_TO_CORPORATE_ID, POSITION_TO_DURATION_OF_ACTIVITY, POSITION_TO_POSITION, POSITION_TO_COMPANY, POSITION_TO_CEO_AUTHORITY, COMPANY_TO_CORPORATE_ID, COMPANY_TO_POSITION, COMPANY_TO_CORPORATE_REGISTRATION_NUMBER, COMPANY_TO_DATE, COMPANY_TO_ADDRESS, COMPANY_TO_SUBJECT_OF_ACTIVITY, COMPANY_TO_DURATION_OF_ACTIVITY, COMPANY_TO_PERSONAL_ID, COMPANY_TO_PERSON, COMPANY_TO_COMPANY, COMPANY_TO_COMPANY_CAPITAL, COMPANY_TO_CEO_AUTHORITY, SUBJECT_OF_ACTIVITY_TO_DATE, SUBJECT_OF_ACTIVITY_TO_DURATION_OF_ACTIVITY, DURATION_OF_ACTIVITY_TO_DATE, DURATION_OF_ACTIVITY_TO_POSITION.

Text:
{text}"""

PHASE2_USER_INSTRUCTION = """Compare the official gazette text with its extracted JSON. Audit and fix the following:
1. Merge fragmented addresses into a single unified ADDRESS entity. Delete separated sub-address components.
2. Remove any relation connecting to or from a SIGN_RULE entity.
3. Verify that all extracted entities are exact substrings of the Original Text. Delete any hallucinated entities or relations linking to them.
4. Correct any relation mapped incorrectly.
5. Ensure that 11-digit numbers (like 10360031006) are classified strictly as `CORPORATE_ID`.
6. Strictly use ONLY the 26 allowed relation types in the schema.

Original Text:
{text}

Current JSON to audit:
{json_data}"""

# ==========================================
# توابع کمکی پارسر و نرمال‌سازی داده‌ها
# ==========================================
def clean_legal_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'#x0[DdAa];', ' ', text)
    bidi_chars_pattern = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')
    text = bidi_chars_pattern.sub('', text)
    text = text.replace('\u0643', '\u06a9').replace('\u064a', '\u06cc')
    return " ".join(text.split())

def extract_json_from_text(text: str) -> dict:
    text_clean = text.strip()
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text_clean, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r'```\s*(.*?)\s*```', text_clean, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start_idx = text_clean.find('{')
        end_idx = text_clean.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text_clean[start_idx:end_idx+1])
        return json.loads(text_clean)
    except Exception as e:
        raise ValueError(f"قالب متنی مدل قابلیت تبدیل به JSON را نداشت.\nخطا: {e}")

def normalize_llm_hallucinations(corrected_json: dict, original_text: str = "") -> dict:
    entities = corrected_json.get("entities", [])
    relations = corrected_json.get("relations", [])
    clean_orig = clean_legal_text(original_text)

    entity_type_correction = {
        "COMPANY_REGISTRATION_NUMBER": "CORPORATE_REGISTRATION_NUMBER",
        "REGISTRATION_NUMBER": "CORPORATE_REGISTRATION_NUMBER",
        "COMPANY_ID": "CORPORATE_ID",
        "NATIONAL_ID": "CORPORATE_ID"
    }
    
    sanitized_entities = []
    valid_entity_texts = set()

    for ent in entities:
        ent_text = ent.get("text") or ent.get("name") or ""
        ent_type = ent.get("type", "")
        ent_text_clean = clean_legal_text(ent_text)
        if not ent_text_clean: continue

        if clean_orig and (ent_text_clean not in clean_orig): continue
        if ent_type in entity_type_correction: ent_type = entity_type_correction[ent_type]

        digits_only = re.sub(r'\D', '', ent_text_clean)
        if len(digits_only) == 11 and ent_type in ["COMPANY_CAPITAL", "PERSONAL_ID"]:
            if not any(word in ent_text_clean for word in ["ریال", "تومان", "ریالی"]):
                ent_type = "CORPORATE_ID"
        
        sanitized_entities.append({"text": ent_text_clean, "type": ent_type})
        valid_entity_texts.add(ent_text_clean)
    
    corrected_json["entities"] = sanitized_entities
    entity_types = {ent["text"]: ent["type"] for ent in sanitized_entities}
    sanitized_relations = []

    VALID_RELATION_MAPS = {
        "PERSON_TO_POSITION": [("PERSON", "POSITION")],
        "PERSON_TO_PERSONAL_ID": [("PERSON", "PERSONAL_ID")],
        "PERSON_TO_COMPANY": [("PERSON", "COMPANY")],
        "PERSON_TO_DURATION_OF_ACTIVITY": [("PERSON", "DURATION_OF_ACTIVITY")],
        "PERSON_TO_PERSON": [("PERSON", "PERSON")],
        "PERSON_TO_CORPORATE_ID": [("PERSON", "CORPORATE_ID")],
        "POSITION_TO_DURATION_OF_ACTIVITY": [("POSITION", "DURATION_OF_ACTIVITY")],
        "POSITION_TO_POSITION": [("POSITION", "POSITION")],
        "POSITION_TO_COMPANY": [("POSITION", "COMPANY")],
        "POSITION_TO_CEO_AUTHORITY": [("POSITION", "CEO_AUTHORITY")],
        "COMPANY_TO_CORPORATE_ID": [("COMPANY", "CORPORATE_ID")],
        "COMPANY_TO_POSITION": [("COMPANY", "POSITION")],
        "COMPANY_TO_CORPORATE_REGISTRATION_NUMBER": [("COMPANY", "CORPORATE_REGISTRATION_NUMBER")],
        "COMPANY_TO_DATE": [("COMPANY", "DATE")],
        "COMPANY_TO_ADDRESS": [("COMPANY", "ADDRESS")],
        "COMPANY_TO_SUBJECT_OF_ACTIVITY": [("COMPANY", "SUBJECT_OF_ACTIVITY")],
        "COMPANY_TO_DURATION_OF_ACTIVITY": [("COMPANY", "DURATION_OF_ACTIVITY")],
        "COMPANY_TO_PERSONAL_ID": [("COMPANY", "PERSONAL_ID")],
        "COMPANY_TO_PERSON": [("COMPANY", "PERSON")],
        "COMPANY_TO_COMPANY": [("COMPANY", "COMPANY")],
        "COMPANY_TO_COMPANY_CAPITAL": [("COMPANY", "COMPANY_CAPITAL")],
        "COMPANY_TO_CEO_AUTHORITY": [("COMPANY", "CEO_AUTHORITY")],
        "SUBJECT_OF_ACTIVITY_TO_DATE": [("SUBJECT_OF_ACTIVITY", "DATE")],
        "SUBJECT_OF_ACTIVITY_TO_DURATION_OF_ACTIVITY": [("SUBJECT_OF_ACTIVITY", "DURATION_OF_ACTIVITY")],
        "DURATION_OF_ACTIVITY_TO_DATE": [("DURATION_OF_ACTIVITY", "DATE")],
        "DURATION_OF_ACTIVITY_TO_POSITION": [("DURATION_OF_ACTIVITY", "POSITION")]
    }

    for rel in relations:
        from_ent = rel.get("from_entity") or rel.get("source") or ""
        to_ent = rel.get("to_entity") or rel.get("target") or ""
        rel_type = rel.get("relation_type") or rel.get("type") or ""

        from_clean = clean_legal_text(from_ent)
        to_clean = clean_legal_text(to_ent)

        if from_clean not in valid_entity_texts or to_clean not in valid_entity_texts: continue

        from_type = entity_types[from_clean]
        to_type = entity_types[to_clean]

        if rel_type in VALID_RELATION_MAPS:
            if (from_type, to_type) in VALID_RELATION_MAPS[rel_type]:
                sanitized_relations.append({
                    "from_entity": from_clean,
                    "to_entity": to_clean,
                    "relation_type": rel_type
                })

    corrected_json["relations"] = sanitized_relations
    return corrected_json

# ==========================================
# ریسه‌های کارگر فاز ۱ (Google Gemini API)
# ==========================================
def phase1_worker(thread_id, api_key, df, pbar):
    client = genai.Client(api_key=api_key)

    while True:
        try:
            idx = phase1_queue.get_nowait()
        except queue.Empty:
            break

        text = df['cleaned_text'].iloc[idx]
        prompt = PHASE1_USER_INSTRUCTION.format(text=text)

        while True:
            try:
                # فراخوانی رسمی API جمینای با لایبرری جدید و خروجی اجباری JSON
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                
                raw_json = extract_json_from_text(response.text)
                raw_json = normalize_llm_hallucinations(raw_json, text)
                raw_json["row_index"] = idx

                with phase1_lock:
                    phase1_shared_results.append(raw_json)
                    with open(PHASE1_OUT, 'w', encoding='utf-8') as f:
                        json.dump(phase1_shared_results, f, ensure_ascii=False, indent=2)

                if RUN_PHASE_2:
                    phase2_queue.put(raw_json)
                break

            except Exception as e:
                tqdm.write(f"⚠️ [Thread-{thread_id} Gemini P1] خطا در سطر {idx}: {e}. تلاش مجدد تا ۵ ثانیه دیگر...")
                time.sleep(5)

        pbar.update(1)
        phase1_queue.task_done()

# ==========================================
# ریسه‌های کارگر فاز ۲ (Google Gemini API Audit)
# ==========================================
def phase2_worker(thread_id, api_key, df, pbar):
    client = genai.Client(api_key=api_key)

    while True:
        entry = phase2_queue.get()
        if entry is None:
            phase2_queue.task_done()
            break

        idx = entry.get("row_index")
        text = df['cleaned_text'].iloc[idx]
        prompt = PHASE2_USER_INSTRUCTION.format(text=text, json_data=json.dumps(entry, ensure_ascii=False, indent=2))

        while True:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                
                corrected_json = extract_json_from_text(response.text)
                corrected_json = normalize_llm_hallucinations(corrected_json, text)
                corrected_json["row_index"] = idx

                with phase2_lock:
                    phase2_shared_results.append(corrected_json)
                    with open(PHASE2_OUT, 'w', encoding='utf-8') as f:
                        json.dump(phase2_shared_results, f, ensure_ascii=False, indent=2)
                break

            except Exception as e:
                tqdm.write(f"⚠️ [Thread-{thread_id} Gemini P2] خطا در ممیزی سطر {idx}: {e}. تلاش مجدد تا ۵ ثانیه دیگر...")
                time.sleep(5)

        pbar.update(1)
        phase2_queue.task_done()

# ==========================================
# فاز ۳: هم‌ترازی محلی توکن‌ها برای پارس‌برت
# ==========================================
def run_dataset_alignment(df):
    print("⏳ شروع فاز ۳ (محلی): هم‌ترازی توکن‌ها با ParsBERT...")

    if not os.path.exists(PHASE2_OUT):
        raise FileNotFoundError(f"فایل دیتای ممیزی‌شده یافت نشد: '{PHASE2_OUT}'.")

    with open(PHASE2_OUT, 'r', encoding='utf-8') as f:
        corrected_data = json.load(f)

    label_list = ["O"]
    for t in VALID_ENTITY_TYPES:
        label_list.extend([f"B-{t}", f"I-{t}"])

    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, ensure_ascii=False, indent=2)

    from transformers import BertTokenizerFast
    from datasets import Dataset
    tokenizer = BertTokenizerFast.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    processed_list = []

    for entry in tqdm(corrected_data):
        if "entities" not in entry or len(entry["entities"]) == 0: continue
        idx = entry["row_index"]
        if idx >= len(df): continue

        text = df['cleaned_text'].iloc[idx]
        unique_entities = []
        seen = set()
        for ent in entry["entities"]:
            e_text = ent.get("text")
            e_type = ent.get("type")
            if e_text and e_type and (e_text, e_type) not in seen:
                seen.add((e_text, e_type))
                unique_entities.append(ent)

        tokenized_input = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)
        labels = [label_to_id["O"]] * len(tokenized_input["input_ids"])
        offsets = tokenized_input["offset_mapping"]

        sorted_entities = sorted(unique_entities, key=lambda x: len(str(x.get("text") or "")), reverse=True)

        for ent in sorted_entities:
            ent_text = clean_legal_text(str(ent.get("text") or ""))
            ent_type = ent.get("type")

            if not ent_text or f"B-{ent_type}" not in label_to_id: continue

            pattern = rf"(?<!\w){re.escape(ent_text)}(?!\w)"
            matches = [m.span() for m in re.finditer(pattern, text)]

            for start_char, end_char in matches:
                first_token = True
                for t_idx, (start, end) in enumerate(offsets):
                    if start == end:
                        labels[t_idx] = -100
                        continue
                    if start_char <= start < end_char:
                        if first_token:
                            labels[t_idx] = label_to_id[f"B-{ent_type}"]
                            first_token = False
                        else:
                            if labels[t_idx] == label_to_id["O"]:
                                labels[t_idx] = label_to_id[f"I-{ent_type}"]

        tokenized_input["labels"] = labels
        tokenized_input.pop("offset_mapping")
        processed_list.append(tokenized_input)

    final_dataset = Dataset.from_list(processed_list)
    final_dataset.save_to_disk(OUTPUT_DATASET_DIR)
    print(f"🎉 فرآیند هم‌ترازی با موفقیت پایان یافت! دیتای نهایی در '{OUTPUT_DATASET_DIR}' ذخیره شد.")

# ==========================================
# ارکستراتور و مدیریت صف‌ها
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"فایل اکسل اصلی در مسیر '{EXCEL_PATH}' یافت نشد.")

    print("📖 در حال بارگذاری اکسل...")
    df_excel = pd.read_excel(EXCEL_PATH)
    df_excel['cleaned_text'] = df_excel['NewsText'].apply(clean_legal_text)

    if os.path.exists(PHASE1_OUT):
        with open(PHASE1_OUT, 'r', encoding='utf-8') as f:
            phase1_shared_results = json.load(f)
    if os.path.exists(PHASE2_OUT):
        with open(PHASE2_OUT, 'r', encoding='utf-8') as f:
            phase2_shared_results = json.load(f)

    p1_done_indices = {entry["row_index"] for entry in phase1_shared_results if "row_index" in entry}
    p2_done_indices = {entry["row_index"] for entry in phase2_shared_results if "row_index" in entry}

    # پر کردن صف فاز ۱ (بر اساس بچ انتخابی Active Learning در صورت وجود)
    p1_added = 0
    if os.path.exists(SELECTED_BATCH_PATH):
        print(f"🎯 بارگذاری داده‌های هدفمند از فایل '{SELECTED_BATCH_PATH}'...")
        with open(SELECTED_BATCH_PATH, 'r', encoding='utf-8') as f:
            selected_indices = json.load(f)
        for idx in selected_indices:
            if p1_added >= PHASE_1_BATCH_LIMIT: break
            if idx not in p1_done_indices and idx < len(df_excel):
                phase1_queue.put(idx)
                p1_added += 1
    else:
        print("📌 بارگذاری ترتیبی سطر‌ها از فایل اکسل...")
        for idx, row in df_excel.iterrows():
            if p1_added >= PHASE_1_BATCH_LIMIT: break
            if row['cleaned_text'] and idx not in p1_done_indices:
                phase1_queue.put(idx)
                p1_added += 1

    # پر کردن صف فاز ۲
    for entry in phase1_shared_results:
        idx = entry.get("row_index")
        if idx is not None and idx not in p2_done_indices:
            phase2_queue.put(entry)

    total_p1_tasks = phase1_queue.qsize()
    total_p2_tasks = phase2_queue.qsize() + (total_p1_tasks if RUN_PHASE_2 else 0)

    p1_pbar = tqdm(total=total_p1_tasks, desc="📈 پیشرفت فاز ۱ Gemini (استخراج)")
    p2_pbar = tqdm(total=total_p2_tasks, desc="🔍 پیشرفت فاز ۲ Gemini (ممیزی)")

    p1_threads = []
    p2_threads = []

    # ساخت ریسه‌های ۲ به ۲ فاز ۱ و فاز ۲
    if RUN_PHASE_1 and total_p1_tasks > 0:
        for i in range(NUM_PHASE_1_THREADS):
            api_key = GEMINI_API_KEYS[i % len(GEMINI_API_KEYS)]
            t = threading.Thread(target=phase1_worker, args=(i, api_key, df_excel, p1_pbar))
            t.start()
            p1_threads.append(t)

    if RUN_PHASE_2:
        for i in range(NUM_PHASE_2_THREADS):
            api_key = GEMINI_API_KEYS[(i + NUM_PHASE_1_THREADS) % len(GEMINI_API_KEYS)]
            t = threading.Thread(target=phase2_worker, args=(i, api_key, df_excel, p2_pbar))
            t.start()
            p2_threads.append(t)

    for t in p1_threads: t.join()
    p1_pbar.close()

    if RUN_PHASE_2:
        for _ in range(NUM_PHASE_2_THREADS):
            phase2_queue.put(None)
        for t in p2_threads: t.join()
    p2_pbar.close()

    if RUN_PHASE_3:
        run_dataset_alignment(df_excel)