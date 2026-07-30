# solve_phase3_production_master.py
# پایپ‌لاین استادانه و نهایی برای استخراج دانش و ساخت پروفایل‌ها جهت ارسال به مسابقه

import json
import os
import re
import pandas as pd
from tqdm import tqdm

# ==========================================
# مسیرهای فایل‌ها (پشتیبانی از هر دو فاز validation و test)
# ==========================================
DATA_DIR = "data"
BASE_EXCEL = os.path.join(DATA_DIR, "base.xlsx")

# بررسی وجود فایل‌های فاز آزمون (test) یا فاز بهبود (validation)
if os.path.exists(os.path.join(DATA_DIR, "test_names.csv")):
    TARGET_EXCEL = os.path.join(DATA_DIR, "test.xlsx")
    TARGET_NAMES_CSV = os.path.join(DATA_DIR, "test_names.csv")
    OUTPUT_JSON_PATH = "submission_test_final.json"
    print("🎯 وضعیت: شناسایی فاز نهایی آزمون (TEST PHASE)")
else:
    TARGET_EXCEL = os.path.join(DATA_DIR, "validation.xlsx")
    TARGET_NAMES_CSV = os.path.join(DATA_DIR, "validation_names.csv")
    if not os.path.exists(TARGET_NAMES_CSV):
        TARGET_NAMES_CSV = os.path.join(
            DATA_DIR, "validation_people_names.csv"
        )
    OUTPUT_JSON_PATH = "submission_validation_perfect.json"
    print("🎯 وضعیت: شناسایی فاز بهبود (VALIDATION PHASE)")

VALID_ATTRIBUTE_KEYS = [
    "POSITION",
    "CEO_AUTHORITY",
    "COMPANY",
    "CORPORATE_ID",
    "CORPORATE_REGISTRATION_NUMBER",
    "COMPANY_CAPITAL",
    "ADDRESS",
    "DATE",
    "DURATION_OF_ACTIVITY",
    "SIGN_RULE",
    "SUBJECT_OF_ACTIVITY",
]

# ==========================================
# کلمات ممنوعه برای حذف False Positive شرکت‌ها
# ==========================================
FORBIDDEN_COMPANY_WORDS = [
    "سهم الشرکه",
    "سهم‌الشرکه",
    "حاصل نگردید",
    "لیست شرکا",
    "لیست شرکاء",
    "مزبور که در تاریخ",
    "تصمیمات ذیل",
    "اداره ثبت",
    "مرجع ثبت",
    "ثبت شرکتها",
    "ثبت شرکت ها",
    "برای مدت",
    "انتقال یافت",
    "واگذار",
    "دفتر ثبت",
    "واحد ثبتی",
    "شرکت ها",
    "شرکت‌ها",
    "پ۹",
    "ش۹",
    "کد پستی",
    "کدپستی",
    "پلاک",
    "طبق صورتجلسه",
    "آگهی تغییرات",
    "سازمان بهزیستی کشور تصمیمات",
]

# ==========================================
# لیست کامل و جامع سمت‌های روزنامه رسمی
# ==========================================
EXPANDED_POSITIONS = [
    "رئیس هیئت مدیره",
    "رئیس هیأت مدیره",
    "رییس هیئت مدیره",
    "رییس هیأت مدیره",
    "نائب رئیس هیئت مدیره",
    "نایب رئیس هیئت مدیره",
    "نائب رئیس هیأت مدیره",
    "نایب رئیس هیأت مدیره",
    "نائب رییس هیئت مدیره",
    "نایب رییس هیئت مدیره",
    "نائب رییس هیأت مدیره",
    "نایب رییس هیأت مدیره",
    "مدیرعامل",
    "مدیر عامل",
    "قائم مقام مدیرعامل",
    "قائم مقام مدیر عامل",
    "عضو هیئت مدیره",
    "عضو هیأت مدیره",
    "عضوهیئت مدیره",
    "عضوهیات مدیره",
    "عضو اصلی هیئت مدیره",
    "عضو اصلی هیات مدیره",
    "عضو علی البدل هیئت مدیره",
    "عضو علی البدل هیات مدیره",
    "عضو علی البدل",
    "بازرس اصلی",
    "بازرس علی البدل",
    "منشی هیئت مدیره",
    "منشی هیأت مدیره",
]

# ==========================================
# الگوهای جامع مدت فعالیت
# ==========================================
DURATION_REGEXES = [
    r"برای\s+مدت\s+(?:دو|۲|سه|۳|پنج|۵|یک|۱|چهار|۴)\s+سال",
    r"برای\s+(?:دو|۲|سه|۳|پنج|۵|یک|۱)\s+سال\s*مالی",
    r"برای\s+مدت\s+یک\s*سال\s+مالی",
    r"برای\s+یک\s*سال\s+مالی",
    r"برای\s+مدت\s+یک\s*سال",
    r"برای\s+یک\s*سال",
    r"برای\s+یکسال\s+مالی",
    r"برای\s+مدت\s+یکسال",
    r"به\s+مدت\s+(?:دو|۲|سه|۳|پنج|۵|یک|۱)\s+سال",
    r"برای\s+مدت\s+نامحدود",
    r"به\s+مدت\s+نامحدود",
    r"برای\s+(?:باقی\s*مانده|بقیه)\s+(?:مدت|دوره)\s+تصدی",
    r"تا\s+پایان\s+مدت\s+تصدی",
    r"تا\s+تاریخ\s+\d{2,4}/\d{1,2}/\d{2,4}",
    r"لغایت\s*\d{2,4}/\d{1,2}/\d{2,4}",
]


def clean_text(text):
    """پاک‌سازی و استانداردسازی پایه متون فارسی"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"#x0[DdAa];", " ", text)
    text = text.replace("\u0643", "\u06a9").replace("\u064a", "\u06cc")
    bidi_pattern = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
    text = bidi_pattern.sub("", text)
    return " ".join(text.split())


def is_valid_iranian_national_id(code: str) -> bool:
    """اعتبارسنجی الگوریتمی باقی‌مانده تقسیم بر ۱۱ (Checksum) کد ملی"""
    if not re.match(r"^\d{10}$", code):
        return False
    if (
        code.startswith("98")
        or code.startswith("139")
        or code.startswith("10")
        or code.startswith("14")
    ):
        return False
    check = int(code[9])
    s = sum(int(code[i]) * (10 - i) for i in range(9)) % 11
    return (s < 2 and check == s) or (s >= 2 and check == 11 - s)


def clean_company_name(comp_name):
    """پاک‌سازی سخت‌گیرانه نام شرکت‌ها و حذف متون حقوقی اضافی (Fix FP=55)"""
    comp_name = comp_name.strip()
    if len(comp_name) > 65 or len(comp_name) < 5:
        return None

    for forbidden in FORBIDDEN_COMPANY_WORDS:
        if forbidden in comp_name:
            return None

    # حذف پسوندهای اشتباه انتخابی
    comp_name = re.sub(r"\s+(?:به|شماره|شناسه|ثبت|به سمت|تعیین|انتخاب)$", "", comp_name)
    return comp_name.strip()


def is_same_person(query_name, doc_text):
    """تطبیق انعطاف‌پذیر نام اشخاص (Fuzzy Substring Matching)"""
    q_clean = clean_text(query_name)
    if q_clean in doc_text:
        return True

    # مقایسه کلمات اصلی اسم بدون پیشوندهای عمومی
    words = [
        w
        for w in q_clean.split()
        if w not in ["آقای", "خانم", "سید", "سیده", "دکتر"]
    ]
    if len(words) >= 2:
        pattern = r"\s+".join([re.escape(w) for w in words])
        if re.search(pattern, doc_text):
            return True
    return False


def build_corpus():
    """ساخت کورپوس مجاز (صرفاً base.xlsx + validation.xlsx / test.xlsx)"""
    all_docs = []
    if os.path.exists(BASE_EXCEL):
        print(f"📖 بارگذاری فایل مرجع پایه: '{BASE_EXCEL}'")
        df_base = pd.read_excel(BASE_EXCEL)
        all_docs.append(df_base)

    if os.path.exists(TARGET_EXCEL):
        print(f"📖 بارگذاری فایل هدف فاز: '{TARGET_EXCEL}'")
        df_target = pd.read_excel(TARGET_EXCEL)
        all_docs.append(df_target)

    if not all_docs:
        raise FileNotFoundError("❌ هیچ‌کدام از فایل‌های base یا target یافت نشدند!")

    df_corpus = pd.concat(all_docs, ignore_index=True)
    df_corpus["clean_text"] = df_corpus["NewsText"].apply(clean_text)
    print(f"✅ کورپوس نهایی مجاز ساخته شد با {len(df_corpus)} سند روزنامه رسمی.")
    return df_corpus


def run_pipeline():
    if not os.path.exists(TARGET_NAMES_CSV):
        raise FileNotFoundError(f"فایل اسامی افراد هدف در '{TARGET_NAMES_CSV}' یافت نشد.")

    df_target_names = pd.read_csv(TARGET_NAMES_CSV)
    df_corpus = build_corpus()

    final_output = []
    print(f"⚡ شروع پردازش و تکمیل پروفایل برای {len(df_target_names)} فرد هدف...")

    for _, q_row in tqdm(df_target_names.iterrows(), total=len(df_target_names)):
        q_id = int(q_row["query_id"])
        target_name = str(q_row["names"]).strip()

        matched_names = set([target_name])
        matched_pids = set()
        matched_attributes = {k: set() for k in VALID_ATTRIBUTE_KEYS}

        for _, doc_row in df_corpus.iterrows():
            text = doc_row["clean_text"]

            if not is_same_person(target_name, text):
                continue

            # ۱. استخراج کد ملی (با لنگراندازی و Checksum)
            pid_patterns = [
                rf"{re.escape(target_name)}[^\d\n\.]*?(?:کدملی|کد ملی|شماره ملی|ش م)[^\d]*?(\d{{10}})",
                rf"(?:آقای|خانم)\s+{re.escape(target_name)}[^\d\n\.]*?(\d{{10}})",
                rf"{re.escape(target_name)}[^\d\n\.]*?(\d{{10}})",
            ]
            for pat in pid_patterns:
                for m in re.findall(pat, text):
                    if is_valid_iranian_national_id(m):
                        matched_pids.add(m)

            # شکستن متن به پاراگراف‌ها یا جملات برای حفظ بافت
            sentences = re.split(r"[\n\.]| و | - | ـ ", text)
            for sent in sentences:
                if target_name in sent or any(w in sent for w in target_name.split() if len(w) > 2):

                    # ۲. استخراج سمت‌ها (POSITION)
                    for pos in EXPANDED_POSITIONS:
                        if pos in sent:
                            matched_attributes["POSITION"].add(pos)

                    # ۳. استخراج مدت فعالیت (DURATION_OF_ACTIVITY)
                    for dur_reg in DURATION_REGEXES:
                        for dm in re.findall(dur_reg, sent):
                            matched_attributes["DURATION_OF_ACTIVITY"].add(dm.strip())

                    # ۴. استخراج پاک‌سازی شده نام شرکت (COMPANY)
                    comp_match = re.search(
                        r"(?:شرکت|موسسه|مؤسسه|سازمان)\s+([^،\.\n]+?)(?=\s+(?:به|شماره|شناسه|ثبت|به سمت|تعیین|انتخاب|$))",
                        sent,
                    )
                    if comp_match:
                        full_comp = comp_match.group(0).strip()
                        cleaned_comp = clean_company_name(full_comp)
                        if cleaned_comp:
                            matched_attributes["COMPANY"].add(cleaned_comp)

                            # استخراج شناسه ملی مرتبط با همین شرکت
                            c_id_match = re.search(
                                rf"{re.escape(cleaned_comp)}[^\d]*?(?:شناسه ملی)\s*(\d{{11}})",
                                text,
                            )
                            if c_id_match:
                                matched_attributes["CORPORATE_ID"].add(c_id_match.group(1))

                    # ۵. استخراج شماره ثبت شرکت
                    reg_match = re.search(r"(?:شماره\s+ثبت|تحت\s+شماره)\s*(\d+)", sent)
                    if reg_match:
                        matched_attributes["CORPORATE_REGISTRATION_NUMBER"].add(reg_match.group(1))

            # ۶. استخراج تاریخ آگهی در صورت حضور فرد
            if matched_pids or matched_attributes["POSITION"] or matched_attributes["COMPANY"]:
                d_matches = re.findall(r"\b\d{2,4}/\d{1,2}/\d{2,4}\b", text)
                matched_attributes["DATE"].update(d_matches)

        # پاک‌سازی کدهای ملی سرایت‌کرده بهزیستی
        SHARED_LEAKED_IDS = {"4859813669", "4969752958"}
        if len(matched_pids) > 1:
            matched_pids = {p for p in matched_pids if p not in SHARED_LEAKED_IDS}

        # فیلتر کردن ویژگی‌های خالی (حذف کلیدهای بدون مقدار طبق قوانین)
        clean_attrs = {}
        for k, v in matched_attributes.items():
            if v:
                clean_attrs[k] = list(v)

        final_output.append(
            {
                "query_id": q_id,
                "profile": {
                    "names": list(matched_names),
                    "personal_ids": list(matched_pids),
                    "attributes": clean_attrs,
                },
            }
        )

    # ذخیره فایل JSON نهایی
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n🎉 فرآیند با موفقیت پایان یافت!")
    print(f"💾 فایل خروجی آماده ارسال در مسیر: '{OUTPUT_JSON_PATH}'")


if __name__ == "__main__":
    run_pipeline()