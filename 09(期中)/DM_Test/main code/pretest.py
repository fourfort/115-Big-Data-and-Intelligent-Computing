# pretest.py（中英雙語保留 + 英文保護 + 高效清理）
import jieba
import pandas as pd
import re
import os
from pathlib import Path

# === 1. 字典設定 ===
jieba.set_dictionary('dict/dict.txt.big')
if Path('dict/movie_dict.txt').exists():
    jieba.load_userdict('dict/movie_dict.txt')

# === 2. 停用詞載入（僅中文）===
STOPWORDS = set([
    '的', '是', '在', '有', '和', '就', '不', '這', '那', '很', '我', '了', 
    '也', '都', '但', '之', '與', '或', '於', '呢', '啊', '吧', '啦', '喔'
])
stopwords_path = "data/cn_stopwords.txt"
if os.path.exists(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            cn_words = [line.strip() for line in f if line.strip()]
            STOPWORDS.update(cn_words)
        print(f"[Success] 載入中文停用詞: {len(STOPWORDS)} 個")
    except Exception as e:
        print(f"[Warning] 停用詞載入失敗: {e}")
else:
    print(f"[Info] 未找到 {stopwords_path}，使用預設中文停用詞")

# === 3. 正規表達式（保留中英文數字）===
# 保留：中文、英文、數字、必要符號（如 - /）
KEEP_PATTERN = re.compile(r'[^\u4e00-\u9fffa-zA-Z0-9\s\-\/]+')
SPACE_PATTERN = re.compile(r'\s+')

# 英文詞彙保護（避免 jieba 切開）
ENGLISH_PROTECT = re.compile(r'\b[a-zA-Z]{2,}\b')  # 至少2個字母的英文詞

class TextPreprocessor:
    def __init__(self, input_file="data/raw_comments.csv"):
        self.input_file = input_file
        self.df = None

    def _load_data(self):
        """安全載入 CSV"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"檔案不存在: {self.input_file}")
        self.df = pd.read_csv(self.input_file, on_bad_lines='skip', dtype={'content': str})
        self.df = self.df.dropna(subset=['content']).reset_index(drop=True)
        print(f"[Success] 原始資料: {len(self.df)} 筆")

    def _clean_text(self, text):
        """保留中英文數字，移除其他符號"""
        text = str(text)
        # 移除非中英文數字的符號（但保留空格、- /）
        text = KEEP_PATTERN.sub(' ', text)
        # 壓縮多餘空格
        text = SPACE_PATTERN.sub(' ', text).strip()
        return text

    def _protect_english(self, text):
        """在 jieba 斷詞前，保護完整英文詞"""
        english_words = ENGLISH_PROTECT.findall(text)
        for word in english_words:
            # 將英文詞替換為帶底線版本，jieba 不會切開
            placeholder = f"EN_{word.upper()}"
            text = text.replace(word, placeholder)
        return text, {placeholder: word for word in english_words}

    def _restore_english(self, tokens, mapping):
        """恢復被保護的英文詞"""
        restored = []
        for token in tokens:
            if token in mapping:
                restored.append(mapping[token])
            else:
                restored.append(token)
        return restored

    def _segment_text(self, text):
        """jieba 斷詞 + 停用詞 + 長度過濾 + 英文保護"""
        if not text.strip():
            return ''

        # Step 1: 保護英文詞
        protected_text, eng_map = self._protect_english(text)

        # Step 2: jieba 斷詞
        words = jieba.cut(protected_text)

        # Step 3: 過濾 + 還原英文
        filtered = []
        for w in words:
            if w in eng_map:
                filtered.append(eng_map[w])  # 直接還原英文
            elif w in STOPWORDS:
                continue
            elif len(w) >= 2:  # 中文至少2字，英文已保護
                filtered.append(w)

        return ' '.join(filtered) if filtered else ''

    def _label_sentiment(self, row):
        """雙重情緒標註：rating + 內容關鍵詞（中英雙語）"""
        rating = row.get('rating')
        content = row.get('clean_content', '')

        # 1. 豆瓣 rating
        if pd.notna(rating):
            try:
                r = float(rating)
                if r >= 4: return 'positive'
                elif r <= 2: return 'negative'
                else: return 'neutral'
            except:
                pass

        # 2. 關鍵詞標註（中英雙語）
        pos_keywords = [
            '好看', '值得推薦', '喜歡', '感動', '精彩', '演技', '神作', 'great', 'amazing',
            'touching', 'brilliant', 'oscar', 'masterpiece', '佩服', '淚流滿面', '熱淚盈眶'
        ]
        neg_keywords = [
            '難看', '失望', '無聊', '爛片', '尷尬', '出戲', '浪費', 'boring', 'disappointing',
            'weird', 'fake', 'not worth', 'pass', '應該', '很假'
        ]

        content_lower = content.lower()
        if any(k in content_lower for k in pos_keywords):
            return 'positive'
        if any(k in content_lower for k in neg_keywords):
            return 'negative'
        return 'neutral'

    def process(self):
        """主流程"""
        self._load_data()

        print("[Processing] 清理文字（保留中英文）...")
        self.df['clean_content'] = self.df['content'].apply(self._clean_text)

        print("[Processing] 斷詞 + 英文保護中...")
        self.df['tokens'] = self.df['clean_content'].apply(self._segment_text)

        print("[Processing] 情緒標註（中英雙語）...")
        self.df['sentiment'] = self.df.apply(self._label_sentiment, axis=1)

        # 移除空 token
        before = len(self.df)
        self.df = self.df[self.df['tokens'] != ''].reset_index(drop=True)
        print(f"[Success] 移除空內容: {before - len(self.df)} 筆")

        # 儲存
        output_path = "data/cleaned_tokens.csv"
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[Success] 前處理完成，輸出 {len(self.df)} 筆 → {output_path}")


        return self.df
        #回傳處理後之數據  