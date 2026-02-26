from cosyvoice.utils.frontend_utils import contains_chinese
import re

def _normalize_year(text):
        """Normalize year formats in text.

        Converts years like 2025, 1999年, 2024/01/01, 2024-01-01 to Chinese characters.
        e.g., 2025 -> 二零二五, 1999年 -> 一九九九年
        """
        digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }

        def year_to_chinese_with_suffix(match):
            year_str = match.group(1)
            chinese_year = ''.join(digit_map[d] for d in year_str)
            suffix = match.group(2) if len(match.groups()) >= 2 else ''
            return chinese_year + suffix

        def year_to_chinese(match):
            year_str = match.group(0)
            return ''.join(digit_map[d] for d in year_str)

        # Match year patterns (order matters):
        # 1. yyyy年 or yyyy 年 (with optional space before 年)
        text = re.sub(r'((?:19|20)\d{2})(\s*年)', year_to_chinese_with_suffix, text)

        # 2. yyyy in date formats like yyyy/mm/dd or yyyy-mm-dd
        text = re.sub(r'((?:19|20)\d{2})(?=[-/])', year_to_chinese, text)

        # 3. standalone yyyy (be conservative to avoid false positives)
        # Only match 19xx or 20xx, not followed by 年 or date separators
        # text = re.sub(r'(?<![0-9])((?:19|20)\d{2})(?![0-9年/-])', year_to_chinese, text)

        return text

def smartread_text_normalize(text):
    text = text.lower()

    text = text.replace("”", "")
    text = text.replace("“", "")
    if contains_chinese(text):
        
        if "显著" not in text:
            text = text.replace("著", "[zh][ù]")

        # Normalize years before other text normalization
        text = _normalize_year(text)
        text = text.replace("≠", "不等于")
        text = text.replace("≤", "小于等于")
        text = text.replace("≥", "大于等于")
        text = text.replace("≈", "约等于")
        text = text.replace("<", "小于")
        text = text.replace(">", "大于")

        # text = zh_tn_model.normalize(text)
        text = text.replace("\n", "")
        # text = replace_blank(text)
        # text = replace_corner_mark(text)
        # text = text.replace(".", "。")
        # text = text.replace(" - ", "，")
        # text = remove_bracket(text)
        # text = re.sub(r'[，,、]+$', '。', text)

    return text

