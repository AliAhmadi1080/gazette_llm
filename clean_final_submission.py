# clean_final_submission.py (جلاکاری نهایی قبل از ارسال)

import json

INPUT_FILE = "submission_phase3_gold.json"
FINAL_FILE = "submission_phase3_perfect.json"

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# کدهای ملی مدیرعاملان سرایت‌کرده در آگهی‌های جمعی
SHARED_LEAKED_IDS = {"4859813669", "4969752958"}

for record in data:
    pids = record["profile"]["personal_ids"]
    # اگر فرد دارای ۲ کد ملی بود و یکی از آن‌ها کد سرایت‌کرده بود، کد دوم را حذف کن
    if len(pids) > 1:
        clean_pids = [pid for pid in pids if pid not in SHARED_LEAKED_IDS]
        if clean_pids:
            record["profile"]["personal_ids"] = clean_pids
        else:
            record["profile"]["personal_ids"] = [pids[0]]

with open(FINAL_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✨ جلاکاری نهایی انجام شد! فایل آماده ارسال: '{FINAL_FILE}'")