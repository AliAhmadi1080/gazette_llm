import os
import re
import json
import queue
import threading
import time
import pandas as pd
from tqdm import tqdm
import anthropic

# ==========================================
# پیکربندی سیستم کنترل دسته‌ای ابری موازی (Parallel Cloud Config)
# ==========================================
RUN_PHASE_1 = False        # فعال‌سازی فاز ۱
RUN_PHASE_2 = False        # فعال‌سازی فاز ۲
RUN_PHASE_3 = True         # فعال‌سازی فاز ۳ (هم‌ترازی لوکال توکن‌ها)

PHASE_1_BATCH_LIMIT = 2000   # تعداد سطر جدید جهت استخراج در این اجرا

# تعیین تعداد ریسه‌ها (توصیه شده: ۲ ریسه برای فاز ۱ و ۲ ریسه برای فاز ۲)
NUM_PHASE_1_THREADS = 5
NUM_PHASE_2_THREADS = 4

# تخصیص کلیدهای API اختصاصی شما (بصورت چرخشی بین ریسه‌ها توزیع می‌شوند)
API_KEYS = [
    "om-CHrPL9qSnLLnUZtD2HXDRCgN4eC6YVNwRNgqUXae",  # کلید اول شما
    "om-ANec1Da5NqjQBBaS6HYKKY27FtvUNmTq92MkB35aq",  # کلید دوم
    "om-2gJ68MXmmPpmbTHhFustUjD1DiNB9deGfFHNDntZf6Y",  # کلید سوم
    'om-EYAPpVB5WHJ494BK8KJKLLYzeomuY9TKGH3yCgkYhRgh',
    'om-5iUGosPSvtcZ4eLXX13S7xU3uaPR3qAQNJGjPYN3',
    'om-u4yDs9spBKKeh91RivqviVi5ZEZiW2uVkBYSyC',
    'om-Lrw99aV2keiH2hDraRCoDipVVq7XTXh5XV9W4bsRt',
    'om-8FezdScu8VWNEpANBr6LtkHqxcwrkGgVJQPDhvNdD',
    'om-8LV1ktRZYruG33y2f34rFDEpwesreAwgJxXYptos',
]

# مسیرهای دایرکتوری داده‌ها
DATA_DIR = "data"
EXCEL_PATH = os.path.join(DATA_DIR, "train-data.xlsx")
PHASE1_OUT = os.path.join(DATA_DIR, "structured_gazette_results.json")
PHASE2_OUT = os.path.join(DATA_DIR, "final_corrected_data.json")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_mapping.json")
OUTPUT_DATASET_DIR = "aligned_ner_dataset"

os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# تعاریف ساختاری و قفل‌های ریسه‌ای (Thread Safety Locks)
# ==========================================
phase1_lock = threading.Lock()
phase2_lock = threading.Lock()

# لیست‌های اشتراکی در حافظه جهت جلوگیری از Disk Read مکرر
phase1_shared_results = []
phase2_shared_results = []

# صف‌های تبادل داده بین ریسه‌ها
phase1_queue = queue.Queue()
phase2_queue = queue.Queue()

# لیست ۱۳ نوع موجودیت مجاز مسابقه
VALID_ENTITY_TYPES = [
    "PERSON", "PERSONAL_ID", "POSITION", "CEO_AUTHORITY",
    "COMPANY", "CORPORATE_ID", "CORPORATE_REGISTRATION_NUMBER",
    "COMPANY_CAPITAL", "ADDRESS", "DATE", "DURATION_OF_ACTIVITY",
    "SIGN_RULE", "SUBJECT_OF_ACTIVITY"
]

# طرح‌واره ساختاریافته جهت اعمال در پارامتر خروجی بومی
PHASE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": VALID_ENTITY_TYPES
                    }
                },
                "required": ["text", "type"],
                "additionalProperties": False
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_entity": {"type": "string"},
                    "to_entity": {"type": "string"},
                    "relation_type": {
                        "type": "string",
                        "enum": [
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
                    }
                },
                "required": ["from_entity", "to_entity", "relation_type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["entities", "relations"],
    "additionalProperties": False
}

# پرامپت استخراج فاز ۱ (بروزرسانی‌شده با قانون شناسه ملی ۱۱ رقمی)
PHASE1_USER_INSTRUCTION = """You must extract legal entities and relations from the text and represent them as a single JSON object.

Strict Rules for Extraction and Graph Consistency:
1. ADDRESS UNIFICATION: Never split an address into multiple entities (such as splitting province, city, street, block, or postal code into separate entities). An address must be extracted as a single, long, unified string in ONE single ADDRESS entity.
2. CORRECT POSTAL CODE CLASSIFICATION: Never classify a 10-digit postal code (کد پستی) as PERSONAL_ID or DATE. Postal codes must be kept inside the ADDRESS text string.
3. SIGN_RULE ISOLATION: The SIGN_RULE entity is strictly isolated. It must NEVER have any relations to other entities.
4. ESTABLISH LOGICAL RELATIONS: If you extract related entities (like COMPANY, SUBJECT_OF_ACTIVITY, CORPORATE_ID, etc.), you MUST establish their logical relations. Do not leave them disconnected with an empty relations array.
5. CORPORATE_ID CLASSIFICATION: Any 11-digit number (usually starting with 10 or 14, such as 10360031006) represents a company's national ID and MUST be classified strictly as `CORPORATE_ID`. NEVER classify 11-digit numbers as `COMPANY_CAPITAL` or `PERSONAL_ID`.
6. JSON KEYS: In "entities" array use only "text" and "type". In "relations" array use only "from_entity", "to_entity" and "relation_type". Never use indices or numeric IDs.

Allowed Entity Types:
PERSON, PERSONAL_ID, POSITION, CEO_AUTHORITY, COMPANY, CORPORATE_ID, CORPORATE_REGISTRATION_NUMBER, COMPANY_CAPITAL, ADDRESS, DATE, DURATION_OF_ACTIVITY, SIGN_RULE, SUBJECT_OF_ACTIVITY.

Allowed Relation Types:
PERSON_TO_POSITION, PERSON_TO_PERSONAL_ID, PERSON_TO_COMPANY, PERSON_TO_DURATION_OF_ACTIVITY, PERSON_TO_PERSON, PERSON_TO_CORPORATE_ID, POSITION_TO_DURATION_OF_ACTIVITY, POSITION_TO_POSITION, POSITION_TO_COMPANY, POSITION_TO_CEO_AUTHORITY, COMPANY_TO_CORPORATE_ID, COMPANY_TO_POSITION, COMPANY_TO_CORPORATE_REGISTRATION_NUMBER, COMPANY_TO_DATE, COMPANY_TO_ADDRESS, COMPANY_TO_SUBJECT_OF_ACTIVITY, COMPANY_TO_DURATION_OF_ACTIVITY, COMPANY_TO_PERSONAL_ID, COMPANY_TO_PERSON, COMPANY_TO_COMPANY, COMPANY_TO_COMPANY_CAPITAL, COMPANY_TO_CEO_AUTHORITY, SUBJECT_OF_ACTIVITY_TO_DATE, SUBJECT_OF_ACTIVITY_TO_DURATION_OF_ACTIVITY, DURATION_OF_ACTIVITY_TO_DATE, DURATION_OF_ACTIVITY_TO_POSITION.

Example of correct JSON output structure:
{{
  "entities": [
    {{"text": "علی اکبر جهانبخت", "type": "PERSON"}},
    {{"text": "0056118384", "type": "PERSONAL_ID"}}
  ],
  "relations": [
    {{"from_entity": "علی اکبر جهانبخت", "to_entity": "0056118384", "relation_type": "PERSON_TO_PERSONAL_ID"}}
  ]
}}

Output ONLY the raw JSON block without markdown formatting or descriptions.

Text:
{text}"""

# پرامپت ممیزی فاز ۲ (بروزرسانی‌شده با قانون شناسه ملی ۱۱ رقمی)
PHASE2_USER_INSTRUCTION = """Compare the official gazette text with its extracted JSON. Audit and fix the following:
1. Merge fragmented addresses into a single unified ADDRESS entity. Delete separated sub-address components.
2. Remove any relation connecting to or from a SIGN_RULE entity.
3. Verify that all extracted entities are exact substrings of the Original Text. Delete any hallucinated entities or relations linking to them.
4. Correct any relation mapped incorrectly (e.g., ensure that POSITION_TO_COMPANY only links a POSITION to a COMPANY, never to a SIGN_RULE).
5. Ensure that 11-digit numbers (like 10360031006) are classified strictly as `CORPORATE_ID`, not as `COMPANY_CAPITAL` or `PERSONAL_ID`.
6. Strictly use ONLY the 26 allowed relation types in the schema.

Original Text:
{text}

Current JSON to audit:
{json_data}"""

# ==========================================
# توابع کمکی پارسر و نرمال‌سازی داده‌ها
# ==========================================


def extract_json_from_text(text: str) -> dict:
    text_clean = text.strip()
    target_str = ""
    
    try:
        # ۱. استخراج بلاک متنی JSON با شرط‌های مختلف
        match = re.search(r'```json\s*(.*?)\s*```', text_clean, re.DOTALL)
        if match:
            target_str = match.group(1)
        else:
            match = re.search(r'```\s*(.*?)\s*```', text_clean, re.DOTALL)
            if match:
                target_str = match.group(1)
            else:
                start_idx = text_clean.find('{')
                end_idx = text_clean.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    target_str = text_clean[start_idx:end_idx+1]
                else:
                    target_str = text_clean

        # ۲. اعمال تصحیح خودکار روی کوتیشن‌های داخلی متون فارسی
        chars = list(target_str.strip())
        in_string = False
        escaped = False
        for i in range(len(chars)):
            c = chars[i]
            if c == '\\' and not escaped:
                escaped = True
                continue
            if c == '"' and not escaped:
                if in_string:
                    # بررسی کاراکترهای بعدی برای تشخیص ساختاری بودن یا نبودن کوتیشن
                    lookahead = "".join(chars[i+1:i+15]).strip()
                    if lookahead.startswith(',') or lookahead.startswith('}') or lookahead.startswith(']') or lookahead.startswith(':'):
                        in_string = False  # کوتیشن ساختاری انتهای رشته
                    else:
                        chars[i] = '\\"'  # کوتیشن داخل متن فارسی (باید اسکیپ شود)
                else:
                    in_string = True  # کوتیشن ساختاری ابتدای رشته
            escaped = False
            
        repaired_json_str = "".join(chars)
        return json.loads(repaired_json_str)
        
    except Exception as e:
        # در صورت شکست دوم، به متد اصلی خود باز می‌گردد تا کرش نکند
        try:
            return json.loads(text_clean)
        except Exception:
            raise ValueError(f"قالب متنی مدل قابلیت تبدیل به JSON را نداشت.\nخطا: {e}")

def get_text_safely(response) -> str:
    text_block = next((block.text for block in response.content if getattr(
        block, 'type', None) == 'text'), None)
    if text_block is None:
        raise ValueError("پاسخ مدل حاوی هیچ بلوک متنی معتبری نبود.")
    return text_block


def clean_legal_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    # ۱. حذف تگ‌های HTML نظیر <br>
    text = re.sub(r'<.*?>', ' ', text)

    # ۲. حذف نویزهای اسکی ناشی از مبدل اکسل نظیر #x0D;
    text = re.sub(r'#x0[DdAa];', ' ', text)

    # ۳. حذف کاراکترهای کنترل جهت یونیکد (BiDi Control Characters) مانند LRM (U+200E) و RLM (U+200F)
    bidi_chars_pattern = re.compile(
        r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')
    text = bidi_chars_pattern.sub('', text)
    

    # ۴. استانداردسازی حروف ی و ک عربی به فارسی
    text = text.replace('\u0643', '\u06a9').replace('\u064a', '\u06cc')

    # ۵. یکپارچه‌سازی فاصله‌ها و حذف فضاهای خالی متوالی
    return " ".join(text.split())


def normalize_llm_hallucinations(corrected_json: dict, original_text: str = "") -> dict:
    """
    موتور ممیزی و تطبیق ساختاری محلی پایتون (Structural Matrix Validation)
    تضمین هم‌ترازی با متن اصلی، حذف روابط غیرمجاز، و فیلتر کدهای ۱۱ رقمی به عنوان شناسه ملی.
    """
    entities = corrected_json.get("entities", [])
    relations = corrected_json.get("relations", [])

    clean_orig = clean_legal_text(original_text)

    # ۱. اصلاح نوع موجودیت‌های نامعتبر یا جایگزین
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
        if not ent_text_clean:
            continue

        # فیلتر محلی پایتون: حذف موجودیت‌هایی که به عنوان توهم خارج از متن اصلی سند ساخته شده‌اند
        if clean_orig and (ent_text_clean not in clean_orig):
            continue

        # اصلاح کلاس‌های نامعتبر اولیه
        if ent_type in entity_type_correction:
            ent_type = entity_type_correction[ent_type]

        # فیلتر فیزیکی سخت‌گیرانه برای شناسه‌های ملی ۱۱ رقمی اشخاص حقوقی
        # اگر عدد استخراج‌شده دقیقاً ۱۱ رقمی باشد ولی کلماتی مانند ریال یا تومان در آن نباشد، تحت کلاس شناسه ملی حقوقی قرار می‌گیرد.
        digits_only = re.sub(r'\D', '', ent_text_clean)
        if len(digits_only) == 11 and ent_type in ["COMPANY_CAPITAL", "PERSONAL_ID"]:
            if not any(word in ent_text_clean for word in ["ریال", "تومان", "ریالی"]):
                ent_type = "CORPORATE_ID"
        
        sanitized_entities.append({
            "text": ent_text_clean,
            "type": ent_type
        })
        valid_entity_texts.add(ent_text_clean)
    
    corrected_json["entities"] = sanitized_entities

    entity_types = {ent["text"]: ent["type"] for ent in sanitized_entities}
    sanitized_relations = []

    # ماتریس اعتبارسنجی جفت‌های مرتب مجاز مبدا و مقصد بر اساس قوانین طرح‌واره مسابقه
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

    relationship_fallback_map = {
        "HAS_ID": "PERSON_TO_PERSONAL_ID",
        "has_national_id": "PERSON_TO_PERSONAL_ID",
        "has_id": "PERSON_TO_PERSONAL_ID",
        "REPRESENTS": "PERSON_TO_COMPANY",
        "represents": "PERSON_TO_COMPANY",
        "is_represented_by": "COMPANY_TO_PERSON",
        "HAS_POSITION": "PERSON_TO_POSITION",
        "has_position": "PERSON_TO_POSITION",
        "has_registration_number": "COMPANY_TO_CORPORATE_REGISTRATION_NUMBER",
        "has_duration": "POSITION_TO_DURATION_OF_ACTIVITY",
        "has_signing_authority": "POSITION_TO_CEO_AUTHORITY",
        "DATE_OF_MEETING": "COMPANY_TO_DATE",
        "REGISTERED_AT": "COMPANY_TO_POSITION",
        "CHANGE_OF_ADDRESS_DATE": "COMPANY_TO_DATE"
    }

    for rel in relations:
        from_ent = rel.get("from_entity") or rel.get("source") or ""
        to_ent = rel.get("to_entity") or rel.get("target") or ""
        rel_type = rel.get("relation_type") or rel.get("type") or ""

        # بازنشانی ایندکس‌های عددی احتمالی مدل به متن واقعی
        if isinstance(from_ent, int) and from_ent < len(entities):
            from_ent = entities[from_ent].get("text") or entities[from_ent].get("name") or ""
        if isinstance(to_ent, int) and to_ent < len(entities):
            to_ent = entities[to_ent].get("text") or entities[to_ent].get("name") or ""

        from_clean = clean_legal_text(from_ent)
        to_clean = clean_legal_text(to_ent)

        # عدم پذیرش رابطه در صورتی که مبدأ یا مقصد آن در فیلتر متن اصلی حذف شده باشند
        if from_clean not in valid_entity_texts or to_clean not in valid_entity_texts:
            continue

        from_type = entity_types[from_clean]
        to_type = entity_types[to_clean]

        # الف) اصلاح اسمی اولیه روابط مترادف
        if rel_type in relationship_fallback_map:
            rel_type = relationship_fallback_map[rel_type]

        # ب) نگاشت منطقی روابط مبهم رایج
        if rel_type in ["has_national_id", "has_id", "PERSON_TO_PERSONAL_ID"]:
            if from_type == "COMPANY" and to_type == "CORPORATE_ID":
                rel_type = "COMPANY_TO_CORPORATE_ID"
            elif from_type == "PERSON" and to_type == "PERSONAL_ID":
                rel_type = "PERSON_TO_PERSONAL_ID"

        if rel_type in ["represents", "PERSON_TO_COMPANY"]:
            if from_type == "COMPANY" and to_type == "PERSON":
                rel_type = "COMPANY_TO_PERSON"
            elif from_type == "PERSON" and to_type == "COMPANY":
                rel_type = "PERSON_TO_COMPANY"

        if rel_type in ["has_position", "PERSON_TO_POSITION"]:
            if from_type == "COMPANY" and to_type == "POSITION":
                rel_type = "COMPANY_TO_POSITION"
            elif from_type == "PERSON" and to_type == "POSITION":
                rel_type = "PERSON_TO_POSITION"

        # ج) فیلترینگ سخت‌گیرانه نوع جفت مرتب (Structural Filter Matrix)
        # هرگز موجودیت‌هایی مانند SIGN_RULE نمی‌توانند وارد رابطه شوند و نوع موجودیت مبدا و مقصد باید کاملاً منطبق باشد
        if rel_type in VALID_RELATION_MAPS:
            allowed_pairs = VALID_RELATION_MAPS[rel_type]
            if (from_type, to_type) not in allowed_pairs:
                continue  # رابطه با ساختار نامنطبق حذف می‌شود
        else:
            continue  # رابطه‌های غیرمجاز حذف می‌شوند

        sanitized_relations.append({
            "from_entity": from_clean,
            "to_entity": to_clean,
            "relation_type": rel_type
        })

    corrected_json["relations"] = sanitized_relations
    return corrected_json

# ==========================================
# کارگران موازی فاز ۱ (Phase 1 Thread Workers)
# ==========================================


def phase1_worker(thread_id, api_key, df, pbar):
    client = anthropic.Anthropic(
        base_url="https://api.openmodel.ai", api_key=api_key)

    while True:
        try:
            idx = phase1_queue.get_nowait()
        except queue.Empty:
            break

        text = df['cleaned_text'].iloc[idx]
        
        # چرخه تلاش مجدد تا رسیدن به خروجی موفقیت‌آمیز
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=16384,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": PHASE1_USER_INSTRUCTION.format(text=text)}
                    ],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": PHASE_JSON_SCHEMA
                        }
                    }
                )
                raw_text = get_text_safely(response)
                raw_json = extract_json_from_text(raw_text)
                
                # ممیزی و فیلتر محلی برای خروجی اولیه فاز ۱
                raw_json = normalize_llm_hallucinations(raw_json, text)
                raw_json["row_index"] = idx

                with phase1_lock:
                    phase1_shared_results.append(raw_json)
                    with open(PHASE1_OUT, 'w', encoding='utf-8') as f:
                        json.dump(phase1_shared_results, f,
                                  ensure_ascii=False, indent=2)

                if RUN_PHASE_2:
                    phase2_queue.put(raw_json)
                
                # خروج از چرخه تکرار در صورت موفقیت کامل
                break

            except Exception as e:
                tqdm.write(f"⚠️ [Thread-{thread_id} P1] خطا در سطر {idx}: {e}. تلاش مجدد تا ۵ ثانیه دیگر...")
                time.sleep(5)

        pbar.update(1)
        phase1_queue.task_done()

# ==========================================
# کارگران موازی فاز ۲ (Phase 2 Thread Workers)
# ==========================================


def phase2_worker(thread_id, api_key, df, pbar):
    client = anthropic.Anthropic(
        base_url="https://api.openmodel.ai", api_key=api_key)

    while True:
        entry = phase2_queue.get()
        if entry is None:
            phase2_queue.task_done()
            break

        idx = entry.get("row_index")
        text = df['cleaned_text'].iloc[idx]

        # چرخه تلاش مجدد تا رسیدن به خروجی موفقیت‌آمیز ممیزی
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=16384,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": PHASE2_USER_INSTRUCTION.format(text=text, json_data=json.dumps(entry, ensure_ascii=False, indent=2))}
                    ],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": PHASE_JSON_SCHEMA
                        }
                    }
                )
                raw_text = get_text_safely(response)
                corrected_json = extract_json_from_text(raw_text)
                
                # ممیزی و فیلتر محلی پایتون
                corrected_json = normalize_llm_hallucinations(corrected_json, text)
                corrected_json["row_index"] = idx

                with phase2_lock:
                    phase2_shared_results.append(corrected_json)
                    with open(PHASE2_OUT, 'w', encoding='utf-8') as f:
                        json.dump(phase2_shared_results, f,
                                  ensure_ascii=False, indent=2)
                
                # خروج از چرخه تکرار در صورت موفقیت کامل
                break

            except Exception as e:
                tqdm.write(f"⚠️ [Thread-{thread_id} P2] خطا در ممیزی سطر {idx}: {e}. تلاش مجدد تا ۵ ثانیه دیگر...")
                time.sleep(5)

        pbar.update(1)
        phase2_queue.task_done()

# ==========================================
# فاز ۳: هم‌ترازی لوکال توکن‌ها
# ==========================================


def run_dataset_alignment(df):
    print("⏳ شروع فاز ۳ (محلی): هم‌ترازی کاراکترها و نشانه‌گذاری توکن‌ها با ParsBERT...")

    if not os.path.exists(PHASE2_OUT):
        raise FileNotFoundError(
            f"فایل دیتای ممیزی شده یافت نشد: '{PHASE2_OUT}'.")

    with open(PHASE2_OUT, 'r', encoding='utf-8') as f:
        corrected_data = json.load(f)

    unique_types = VALID_ENTITY_TYPES
    label_list = ["O"]
    for t in unique_types:
        label_list.extend([f"B-{t}", f"I-{t}"])

    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label},
                  f, ensure_ascii=False, indent=2)

    from transformers import BertTokenizerFast
    from datasets import Dataset
    tokenizer = BertTokenizerFast.from_pretrained(
        "HooshvareLab/bert-fa-zwnj-base")
    processed_list = []

    for entry in tqdm(corrected_data):
        if "entities" not in entry or len(entry["entities"]) == 0:
            continue
        idx = entry["row_index"]
        if idx >= len(df):
            continue

        text = df['cleaned_text'].iloc[idx]

        unique_entities = []
        seen_entities = set()
        for ent in entry["entities"]:
            ent_text = ent.get("text") or ent.get("name") or ent.get("value")
            ent_type = ent.get("type")
            ent_tuple = (ent_text, ent_type)
            if ent_text and ent_type and ent_tuple not in seen_entities:
                seen_entities.add(ent_tuple)
                unique_entities.append(ent)

        tokenized_input = tokenizer(
            text, truncation=True, max_length=512, return_offsets_mapping=True)
        labels = [label_to_id["O"]] * len(tokenized_input["input_ids"])
        offsets = tokenized_input["offset_mapping"]

        sorted_entities = sorted(unique_entities, key=lambda x: len(
            str(x.get("text") or "")), reverse=True)

        for ent in sorted_entities:
            ent_text = str(ent.get("text") or "").strip()
            ent_type = ent.get("type")

            ent_text_clean = clean_legal_text(ent_text)
            if not ent_text_clean or f"B-{ent_type}" not in label_to_id:
                continue

            pattern = rf"(?<!\w){re.escape(ent_text_clean)}(?!\w)"
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
    print(
        f"🎉 کل فرآیند آماده‌سازی با موفقیت پایان یافت! دیتای نهایی در '{OUTPUT_DATASET_DIR}' ذخیره شد.")


# ==========================================
# ارکستراتور و مدیریت ریسه‌ها
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"فایل اکسل اصلی یافت نشد. لطفاً آن را در مسیر '{EXCEL_PATH}' قرار دهید.")

    print("📖 در حال بارگذاری و تمیزکاری اکسل...")
    df_excel = pd.read_excel(EXCEL_PATH)
    df_excel['cleaned_text'] = df_excel['NewsText'].apply(clean_legal_text)

    # ۱. بارگذاری همزمان نتایج قبلی به لیست‌های اشتراکی
    if os.path.exists(PHASE1_OUT):
        with open(PHASE1_OUT, 'r', encoding='utf-8') as f:
            phase1_shared_results = json.load(f)
    if os.path.exists(PHASE2_OUT):
        with open(PHASE2_OUT, 'r', encoding='utf-8') as f:
            phase2_shared_results = json.load(f)

    # استخراج نمایه کاراکترهای پردازش شده قبلی
    p1_done_indices = {entry["row_index"]
                       for entry in phase1_shared_results if "row_index" in entry}
    p2_done_indices = {entry["row_index"]
                       for entry in phase2_shared_results if "row_index" in entry}

    # ۲. راه‌اندازی صف فاز ۱ (محدود به دیتای جدید بر اساس Batch Limit)
    # p1_added = 0
    # for idx, row in df_excel.iterrows():
    #     if p1_added >= PHASE_1_BATCH_LIMIT:
    #         break
    #     if row['cleaned_text'] and idx not in p1_done_indices:
    #         phase1_queue.put(idx)
    #         p1_added += 1
    
    # ---------------selected batch-------------------

    BATCH_INDICES_PATH = os.path.join(DATA_DIR, "selected_batch_indices.json")
    if os.path.exists(BATCH_INDICES_PATH):
        with open(BATCH_INDICES_PATH, 'r', encoding='utf-8') as f:
            selected_indices = json.load(f)
        
        p1_added = 0
        for idx in selected_indices:
            if p1_added >= PHASE_1_BATCH_LIMIT:
                break
            if idx not in p1_done_indices:
                phase1_queue.put(idx)
                p1_added += 1
        print(f"✅ صف فاز ۱ با موفقیت با {p1_added} ردیف فوق‌العاده باارزش از بچ طلاییِ بدون تداخل پر شد.")
    else:
        print("⚠️ بچ طلایی یافت نشد؛ در حال سابمیت ترتیبی عادی...")

    # ---------------selected batch-------------------

    # ۳. راه‌اندازی صف فاز ۲ (شامل فایل‌هایی که فاز ۱ آن‌ها انجام شده اما فاز ۲ مانده است)
    p2_added = 0
    for entry in phase1_shared_results:
        idx = entry.get("row_index")
        if idx is not None and idx not in p2_done_indices:
            phase2_queue.put(entry)
            p2_added += 1

    total_p1_tasks = phase1_queue.qsize()
    total_p2_tasks = phase2_queue.qsize(
    ) + total_p1_tasks if RUN_PHASE_2 else phase2_queue.qsize()

    # ۴. اجرای ریسه‌های موازی
    p1_threads = []
    p2_threads = []

    p1_pbar = tqdm(total=total_p1_tasks, desc="📈 پیشرفت فاز ۱ (استخراج)")
    p2_pbar = tqdm(total=total_p2_tasks, desc="🔍 پیشرفت فاز ۲ (ممیزی)")

    # ساخت ریسه‌های تولیدکننده فاز ۱
    if RUN_PHASE_1 and total_p1_tasks > 0:
        for i in range(NUM_PHASE_1_THREADS):
            api_key = API_KEYS[i % len(API_KEYS)]
            t = threading.Thread(target=phase1_worker, args=(
                i, api_key, df_excel, p1_pbar))
            t.start()
            p1_threads.append(t)

    # ساخت ریسه‌های مصرف‌کننده فاز ۲
    if RUN_PHASE_2:
        for i in range(NUM_PHASE_2_THREADS):
            api_key = API_KEYS[(i + NUM_PHASE_1_THREADS) % len(API_KEYS)]
            t = threading.Thread(target=phase2_worker, args=(
                i, api_key, df_excel, p2_pbar))
            t.start()
            p2_threads.append(t)

    # انتظار برای پایان کار ریسه‌های فاز ۱
    for t in p1_threads:
        t.join()
    p1_pbar.close()

    # ارسال سیگنال اتمام کار به ریسه‌های فاز ۲
    if RUN_PHASE_2:
        for _ in range(NUM_PHASE_2_THREADS):
            phase2_queue.put(None)
        for t in p2_threads:
            t.join()
    p2_pbar.close()

    # ۵. اجرای هم‌ترازی نهایی توکن‌ها بصورت لوکال
    if RUN_PHASE_3:
        run_dataset_alignment(df_excel)