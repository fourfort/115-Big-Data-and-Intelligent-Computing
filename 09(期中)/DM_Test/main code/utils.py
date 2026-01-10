# utils.py（最終優化版：中文 + 排除詞 + 排序）
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil

# ==================== 中文字型強制設定 ====================
def _setup_chinese_font():
    cache_dir = matplotlib.get_cachedir()
    if os.path.exists(cache_dir):
        try: shutil.rmtree(cache_dir)
        except: pass
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()
# ========================================================

# === 動態字型偵測 ===
def _get_font_path():
    candidates = [
        "C:/Windows/Fonts/msjh.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

# === 1. TF-IDF 關鍵詞 ===
def extract_tfidf_keywords(df, top_n=15):
    if 'tokens' not in df.columns:
        raise ValueError("DataFrame 缺少 'tokens' 欄位")
    
    texts = df['tokens'].fillna('')
    if texts.empty:
        return pd.DataFrame(columns=['keyword', 'tfidf_score'])

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'\S+',
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.mean(axis=0).A1
        keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:top_n]
        result = pd.DataFrame(keywords, columns=['keyword', 'tfidf_score'])
        result['tfidf_score'] = result['tfidf_score'].round(6)
        return result
    except:
        return pd.DataFrame(columns=['keyword', 'tfidf_score'])

# === 2. 詞彙關聯 ===
def find_word_associations(df, target_word, min_corr=0.2):
    if 'tokens' not in df.columns:
        raise ValueError("DataFrame 缺少 'tokens' 欄位")
    
    texts = df['tokens'].fillna('')
    if target_word not in ' '.join(texts):
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'\S+', binary=True, min_df=2)
    try:
        dtm = vectorizer.fit_transform(texts)
        dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
        if target_word not in dtm_df.columns:
            return pd.DataFrame()
        correlations = dtm_df.corrwith(dtm_df[target_word])
        assoc = correlations[correlations > min_corr].sort_values(ascending=False)
        result = assoc.drop(target_word).head(10).to_frame('correlation').round(3)
        return result
    except:
        return pd.DataFrame()

# === 3. 詞雲生成===
def generate_wordcloud(df, sentiment=None, min_freq=5, max_freq=1000):
    if 'tokens' not in df.columns:
        raise ValueError("DataFrame 缺少 'tokens' 欄位")
    if sentiment:
        texts = df[df['sentiment'] == sentiment]['tokens']
    else:
        texts = df['tokens']
    
    texts = texts.fillna('')
    if texts.empty:
        print(f"[Warning] No data for sentiment: {sentiment}")
        return
    all_words = []
    for t in texts:
        all_words.extend(t.split())
    
    word_freq = pd.Series(all_words).value_counts()
    word_freq = word_freq[(word_freq >= min_freq) & (word_freq <= max_freq)]
    # 關鍵：排除「電影」、「導演」
    exclude_words = {'電影', '導演'}
    word_freq = word_freq.drop(labels=exclude_words, errors='ignore')
    if word_freq.empty:
        print(f"[Warning] No words after filtering.")
        return
    font_path = _get_font_path()
    wc = WordCloud( font_path=font_path,width=1200, height=800,background_color='white',max_words=120,colormap='viridis',contour_width=1,
        contour_color='steelblue',random_state=42).generate_from_frequencies(word_freq)
    plt.figure(figsize=(14, 9))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'wordcloud for all items', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs("charts", exist_ok=True)
    output_path = f"charts/wordcloud_{'all' if sentiment is None else sentiment}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Success] Word Cloud saved: {output_path}")