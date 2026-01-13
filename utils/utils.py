import re

def split_into_sentences(text: str) -> list[str]:
    """
    将文本按 。 ， ！ ？ 分割 并返回一个列表
    """
    parts = re.split(r'(?<=[。！？!?])', text)  # 保留标点
    sentences = []
    buffer = ''
    for item in parts:
        if item.strip():
            buffer += item.strip()
            if len(buffer) > 35:
                sentences.append(buffer)
                buffer = ''
    if buffer:
        # 判断buffer 中有中文
        if re.search(r'[\u4e00-\u9fff]', buffer):
            sentences.append(buffer)
        elif len(buffer) > 10:  # 英文句子长度阈值
            sentences.append(buffer)
    return sentences
