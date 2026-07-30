# precision_booster.py

import torch
import torch.nn.functional as F
import numpy as np

class NERPrecisionBooster:
    def __init__(self, id_to_label, confidence_threshold=0.75):
        """
        id_to_label: دیکشنری نگاشت شناسه به نام لیبل (id_to_label از فایل label_mapping.json)
        confidence_threshold: آستانه اطمینان برای پذیرش برچسب‌های غیر از O (بین 0.0 تا 1.0)
        """
        self.id_to_label = {int(k): v for k, v in id_to_label.items()}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.confidence_threshold = confidence_threshold
        self.o_class_id = self.label_to_id.get("O", 0)

    def apply_confidence_threshold(self, logits):
        """
        ترفند اول: تنظیم آستانه اطمینان (Probability Calibration)
        اگر احتمال تخصیص یک کلاس به غیر از O کمتر از threshold باشد، آن را به O تبدیل می‌کند.
        """
        # تبدیل خروجی خام مدل (Logits) به توزیع احتمالاتی Softmax
        probabilities = F.softmax(logits, dim=-1) # shape: (batch_size, sequence_length, num_classes)
        
        # پیدا کردن کلاس با بالاترین احتمال و خود مقدار احتمال
        max_probs, preds = torch.max(probabilities, dim=-1)
        
        # آرایه‌ای برای ذخیره پیش‌بینی‌های اصلاح شده
        calibrated_preds = preds.clone()
        
        # ماسک کردن توکن‌هایی که کلاس غیر O دارند اما احتمال آن‌ها زیر آستانه است
        is_not_o = (preds != self.o_class_id)
        under_threshold = (max_probs < self.confidence_threshold)
        
        # تبدیل پیش‌بینی‌های ضعیف غیر O به کلاس مطمئن O
        calibrated_preds[is_not_o & under_threshold] = self.o_class_id
        
        return calibrated_preds.cpu().numpy()

    def apply_transition_constraints(self, sequence_ids):
        """
        ترفند دوم: فیلتر قوانین انتقال توکن (Heuristic Transition Smoothing)
        اصلاح توالی‌های غیرممکن ساختاری (مانند شروع مستقیم با I- بدون داشتن B- قبلی)
        """
        fixed_sequence_ids = list(sequence_ids)
        prev_tag = "O"
        
        for i, class_id in enumerate(fixed_sequence_ids):
            label = self.id_to_label.get(class_id, "O")
            
            if label.startswith("I-"):
                current_entity = label.split("-")[1]
                
                # قانون ۱: اگر برچسب فعلی I-XYZ باشد اما قبل از آن O بوده باشد (شروع غیرقانونی)
                if prev_tag == "O":
                    # اصلاح به B-XYZ جهت تصحیح گرامری
                    target_b_label = f"B-{current_entity}"
                    fixed_sequence_ids[i] = self.label_to_id.get(target_b_label, class_id)
                    label = target_b_label
                
                # قانون ۲: اگر برچسب فعلی I-XYZ باشد اما قبل از آن B-ABC یا I-ABC بوده باشد (تغییر موجودیت بدون B)
                elif prev_tag != "O" and prev_tag.split("-")[1] != current_entity:
                    # برای امنیت بالا و حفظ Precision، آن را به عنوان یک موجودیت جدید B-XYZ شروع کن
                    target_b_label = f"B-{current_entity}"
                    fixed_sequence_ids[i] = self.label_to_id.get(target_b_label, class_id)
                    label = target_b_label
            
            prev_tag = label
            
        return fixed_sequence_ids

    def process_predictions(self, logits):
        """
        اجرای پایپ‌لاین پس‌پردازش روی خروجی مدل (Logits)
        """
        # ۱. اعمال آستانه اطمینان روی احتمالات
        calibrated_batch = self.apply_confidence_threshold(logits)
        
        final_batch_predictions = []
        # ۲. اعمال قوانین گرامری روی تک‌تک جملات بچ
        for sequence in calibrated_batch:
            fixed_sequence = self.apply_transition_constraints(sequence)
            final_batch_predictions.append(fixed_sequence)
            
        return final_batch_predictions