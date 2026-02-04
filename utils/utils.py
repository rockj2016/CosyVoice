import re

def split_into_sentences(text: str, max_length: int = 50) -> list[str]:
    """
    将文本按标点符号分割成句子列表
    支持：。，！？、；：及对应的英文标点
    
    Args:
        text: 要分割的文本
        max_length: 单个句子的最大长度，超过会强制分割
    """
    # 按多种标点分割，包括逗号、句号、感叹号、问号、顿号、分号、冒号、换行符
    parts = re.split(r'(?<=[。！？!?，,、；;：:\n])', text)
    
    sentences = []
    buffer = ''
    
    for item in parts:
        item = item.strip()
        if not item:
            continue
            
        # 如果添加这个item会超过最大长度，先保存buffer
        if buffer and len(buffer) + len(item) > max_length:
            if len(buffer) > 10:  # 最小长度阈值
                sentences.append(buffer)
            buffer = item
        else:
            buffer += item
            
        # 如果buffer本身就超过最大长度，需要强制分割
        while len(buffer) > max_length:
            # 尝试在合适的位置分割（优先在标点处）
            split_pos = max_length
            for punct in ['。', '！', '？', '，', ',', '、', '；', '：', '!', '?', ';', ':', ' ']:
                pos = buffer.rfind(punct, 0, max_length)
                if pos > max_length // 2:  # 至少在中间之后
                    split_pos = pos + 1
                    break
            
            sentences.append(buffer[:split_pos])
            buffer = buffer[split_pos:].strip()
    
    # 处理最后的buffer
    if buffer:
        # 判断buffer中有中文或英文且长度足够
        if re.search(r'[\u4e00-\u9fff]', buffer) or len(buffer) > 10:
            sentences.append(buffer)
    
    return sentences
