import re


def split_into_sentences(text: str, max_length: int = 50) -> list[str]:
    """
    将中文文本按标点符号分割成句子列表
    支持：。，！？、；：及对应的英文标点
    
    Args:
        text: 要分割的文本
        max_length: 单个句子的最大长度，超过会强制分割
    """
    # 按多种标点分割，包括逗号、句号、感叹号、问号、顿号、分号、冒号、换行符
    parts = re.split(r'(?<=[。！？!?，,；;：:])', text)
    
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


def split_into_sentences_en(text: str, max_words: int = 60) -> list[str]:
    """
    Split English text into sentence chunks for TTS.
    
    Splits on sentence-ending punctuation (. ! ?), then merges short sentences
    together up to max_words to avoid overly short TTS segments.
    
    Args:
        text: English text to split
        max_words: Maximum word count per chunk (default 60)
    """
    def _word_count(s: str) -> int:
        return len(s.split())

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split on sentence boundaries, keeping the delimiter attached
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = []
    buffer = ''
    
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        # If adding this sentence exceeds max_words, flush buffer first
        combined = f"{buffer} {sent}" if buffer else sent
        if buffer and _word_count(combined) > max_words:
            sentences.append(buffer.strip())
            buffer = sent
        else:
            buffer = combined
        
        # Force-split if buffer itself is too long
        while _word_count(buffer) > max_words:
            words = buffer.split()
            # Take max_words words, then try to find a sentence/clause boundary nearby
            cut_text = ' '.join(words[:max_words])
            split_pos = len(cut_text)
            for punct in ['. ', '! ', '? ', '; ', ', ']:
                pos = buffer.rfind(punct, 0, split_pos)
                if pos > split_pos // 3:
                    split_pos = pos + len(punct)
                    break
            
            sentences.append(buffer[:split_pos].strip())
            buffer = buffer[split_pos:].strip()
    
    # Flush remaining buffer
    if buffer and _word_count(buffer.strip()) > 2:
        sentences.append(buffer.strip())
    
    return sentences
