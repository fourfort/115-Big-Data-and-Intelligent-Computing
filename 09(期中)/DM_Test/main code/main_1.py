# main.py（最終整合版：圖表頁優化 + 全英文 + 數值標註）
import os
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# --- 1. 圖表風格設定 ---
matplotlib.use('Agg')
sns.set_style("whitegrid")  # 專業白底網格風格

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.unicode_minus': False,
    'figure.figsize': (10, 6),
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

# --- 2. 導入模組 ---
from find import MovieCommentCrawler
from pretest import TextPreprocessor
from miner import TextMiner
from visualizer import ResultVisualizer
from utils import extract_tfidf_keywords, find_word_associations, generate_wordcloud

# --- 3. 主流程 ---
start_time = time.time()
print("=== Text Mining Pipeline ===")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    # Step 1: Crawl
    print("Step 1: Data Collection...")
    crawler = MovieCommentCrawler()
    crawler.crawl_douban()
    crawler.crawl_ptt()
    crawler.crawl_line_today()
    raw_df = crawler.save()
    print(f"Collected: {len(raw_df)} comments\n")

    # Step 2: Preprocess + Tokenize
    print("Step 2: Text Preprocessing & Tokenization...")
    prep = TextPreprocessor()
    df = prep.process()
    print(f"After cleaning: {len(df)} comments\n")

    # Step 3: Mining
    print("Step 3: Text Mining...")
    miner = TextMiner(df)
    miner.classify()
    miner.cluster(n_clusters=3)
    rules = miner.association_rules(min_support=0.02, min_confidence=0.5)
    print("Mining completed\n")

    # Step 4: Visualization (使用 visualizer.py)
    print("Step 4: Visualization (Pie + Cluster + Rules)...")
    viz = ResultVisualizer(df, rules)
    viz.plot_sentiment_pie()
    viz.plot_cluster_sentiment()
    viz.show_rules_table()

    # === Step Step 5: Advanced Analysis
    print("Step 5: Advanced Analysis (TF-IDF + Association + WordCloud)...")

    
    # 5.1 TF-IDF Keywords
    print("   → Extracting top 15 TF-IDF keywords...")
    tfidf_df = extract_tfidf_keywords(df, top_n=15)
    print(tfidf_df.to_string(index=False))
    tfidf_df.to_csv("charts/tfidf_keywords.csv", index=False)
    print("   → Saved: charts/tfidf_keywords.csv")
    plt.figure(figsize=(10, 6))
    plt.bar(tfidf_df['keyword'], tfidf_df['tfidf_score'])
    plt.ylabel("TF-IDF")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_path = "charts/tfidf_keywords.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Check] Saved: {output_path}")
    #

    # 5.2 Word Associations
    print("   → Finding associations with '劉若英' (corr > 0.2)...")
    assoc = find_word_associations(df, target_word="劉若英", min_corr=0.2)
    if not assoc.empty:
        print(assoc.to_string(index=False))
    else:
        print("   No strong associations found.")

    # 5.3 Optimized Word Clouds
    print("   → Generating optimized word clouds...")
    generate_wordcloud(df, sentiment='positive', min_freq=5, max_freq=1000)
    generate_wordcloud(df, sentiment='negative', min_freq=3, max_freq=500)
    generate_wordcloud(df, sentiment=None, min_freq=10, max_freq=2000)

    #新增字詞關聯圖
    '''
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=rules.head(10),   # 前 10 筆
        x="confidence", 
        y="antecedents", 
        hue="consequents",
        palette="viridis"
    )
    plt.title("Top 10 Association Rules (by Confidence)")
    plt.xlabel("Confidence")
    plt.ylabel("Antecedents")
    plt.legend(title="Consequents", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    '''

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=rules.head(10),   # 前 10 筆
        x="confidence", 
        y="antecedents", 
        hue="consequents",
        palette="viridis"
    )
    plt.title("Top 10 Association Rules (by Confidence)")
    plt.xlabel("Confidence")
    plt.ylabel("Antecedents")
    plt.legend(title="Consequents", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 新增儲存與提示
    output_path = "charts/association_rules_scatter.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Check] Saved: {output_path}")
    #2025/11/7
    

    # Step 6: Summary Charts (來源分佈 + 情感圓餅圖已在 visualizer)
    print("Step 6: Summary Charts (Source + Sentiment)...")

    # 6.1 Source Distribution Bar Chart (with value labels)
    plt.figure(figsize=(9, 6))
    ax = sns.countplot(
        data=raw_df, x='source',
        hue='source', palette='Set2', legend=False,
        order=raw_df['source'].value_counts().index
    )
    plt.title('Source Distribution', pad=20)
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # 在每個長條上方標註數量
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=11, fontweight='bold', padding=3)

    plt.tight_layout()
    plt.savefig("charts/source_distribution.png", dpi=200)
    plt.close()
    print("[Check] Saved: charts/source_distribution.png")

    # 6.2 Sentiment Pie (已由 visualizer 處理，僅保留統計)
    sentiment_counts = df['sentiment'].value_counts()
    print(f"\nSummary:")
    print(f"   Total Comments : {len(raw_df)}")
    print(f"   Positive : {sentiment_counts.get('positive',0)} ({sentiment_counts.get('positive',0)/len(df)*100:5.1f}%)")
    print(f"   Negative : {sentiment_counts.get('negative',0)} ({sentiment_counts.get('negative',0)/len(df)*100:5.1f}%)")
    print(f"   Neutral  : {sentiment_counts.get('neutral',0)} ({sentiment_counts.get('neutral',0)/len(df)*100:5.1f}%)")

    # Open charts folder
    print(f"[Open Folder] Opening: {os.path.abspath('charts')}")
    try:
        if os.name == 'nt':  # Windows
            os.startfile("charts")
        elif sys.platform == "darwin":  # macOS
            os.system("open charts")
        else:  # Linux
            os.system("xdg-open charts")
    except Exception as e:
        print(f"[Warning] Failed to open folder: {e}")

except Exception as e:
    print(f"\n[Error] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    duration = time.time() - start_time
    print(f"\n=== Pipeline Completed ===")
    print(f"End Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {duration:.1f}s ({duration/60:.1f} min)")