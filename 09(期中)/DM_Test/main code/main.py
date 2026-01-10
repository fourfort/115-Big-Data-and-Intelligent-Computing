# main.py（表格 + 全詞雲 + 群集修復）
import os
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from find import MovieCommentCrawler
from pretest import TextPreprocessor
from miner import TextMiner
from visualizer import ResultVisualizer
from utils import extract_tfidf_keywords, find_word_associations, generate_wordcloud

# --- 設定 ---
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.unicode_minus': False,
    'figure.figsize': (10, 6),
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
})
warnings.filterwarnings("ignore")

# --- 主流程 ---
start_time = time.time()
print("=== Text Mining Pipeline ===\n")

try:
    # Step 1: Crawl
    print("Step 1: Data Collection...")
    crawler = MovieCommentCrawler()
    crawler.crawl_douban()
    crawler.crawl_ptt()
    crawler.crawl_line_today()
    raw_df = crawler.save()
    print(f"   → Collected: {len(raw_df)} comments\n")

    # Step 2: Preprocess
    print("Step 2: Preprocessing...")
    prep = TextPreprocessor()
    df = prep.process()
    print(f"   → Valid: {len(df)} comments\n")

    # Step 3: Mining
    print("Step 3: Mining...")
    miner = TextMiner(df)
    clf_results = miner.classify()
    cluster_result = miner.cluster(n_clusters=3)  # 這行會寫入 df['cluster']
    # 關鍵：更新 df
    df = miner.df

    # 強制驗證
    # print(f"   → df['cluster'] 欄位存在: {'cluster' in df.columns}")
    print(f"   → cluster 值: {df['cluster'].unique().tolist()}")

    rules = miner.association_rules(min_support=0.02, min_confidence=0.5)

    # Step 4: Visualization
    print("Step 4: Visualization...")
    viz = ResultVisualizer(df, rules)

    # 4.1 情感圓餅圖
    viz.plot_sentiment_pie_final()

    # 4.2 群集主題圖（已修復）
    viz.plot_cluster_sentiment_with_topics()

    # 4.3 關聯規則表格圖
    viz.plot_rules_table()

    viz.show_rules_table()
    # 4.4 分類比較圖
    print("   → Generating classification comparison...")
    viz.plot_classification_comparison(clf_results)

    # Step 5: Advanced Analysis
    print("Step 5: Advanced Analysis...")

    # 5.1 TF-IDF 關鍵詞 → 表格圖 + CSV
    print("   → Extracting top 15 TF-IDF keywords...")
    tfidf_df = extract_tfidf_keywords(df, top_n=15)
    if not tfidf_df.empty:
        tfidf_df['tfidf_score'] = tfidf_df['tfidf_score'].round(6)
        print(tfidf_df.to_string(index=False, float_format='%.6f'))
        tfidf_df.to_csv("charts/tfidf_keywords.csv", index=False)
        viz.plot_tfidf_table(tfidf_df)  # 表格圖
    else:
        print("   [Warning] No TF-IDF keywords.")

    # 5.2 詞彙關聯
    print("   → Finding associations with '劉若英'...")
    assoc = find_word_associations(df, "劉若英", min_corr=0.2)
    if not assoc.empty:
        print(assoc.to_string(index=False))

    # 5.3 詞雲 → 只生成「全詞語」
    print("   → Generating overall word cloud...")
    generate_wordcloud(df, sentiment=None, min_freq=10, max_freq=2000)

    # Step 6: Source Distribution
    print("Step 6: Source Distribution...")
    plt.figure(figsize=(9, 6))
    ax = sns.countplot(data=raw_df, x='source', hue='source',
                       palette='Set2', legend=False,
                       order=raw_df['source'].value_counts().index)
    plt.title('Source Distribution', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Source'); plt.ylabel('Count'); plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("charts/source_distribution.png", dpi=200)
    plt.close()
    print("   → Saved: charts/source_distribution.png")

    # Summary
    counts = df['sentiment'].value_counts()
    print(f"\n=== Final Summary ===")
    print(f"   Total: {len(raw_df):3d} | Valid: {len(df):3d}")
    print(f"   Positive: {counts.get('positive',0):3d} ({counts.get('positive',0)/len(df)*100:5.1f}%)")
    print(f"   Negative: {counts.get('negative',0):3d} ({counts.get('negative',0)/len(df)*100:5.1f}%)")
    print(f"   Neutral : {counts.get('neutral',0):3d} ({counts.get('neutral',0)/len(df)*100:5.1f}%)")

    # Open folder
    print(f"\n[Open Folder] Opening: {os.path.abspath('charts')}")
    try:
        os.startfile("charts") if os.name == 'nt' else \
        os.system("open charts") if sys.platform == "darwin" else \
        os.system("xdg-open charts")
    except: pass

except Exception as e:
    print(f"\n[Error] {e}")
    import traceback; traceback.print_exc()

finally:
    duration = time.time() - start_time
    print(f"\n=== Done in {duration:.1f}s ===")