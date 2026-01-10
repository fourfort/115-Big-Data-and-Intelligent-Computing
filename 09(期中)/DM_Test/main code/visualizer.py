# visualizer.py（最終版：所有中文強制顯示 + 中英分離）
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil

# ==================== 強制中文字型設定 ====================
def _load_chinese_font():
    font_path = "C:/Windows/Fonts/msjh.ttc"
    if os.path.exists(font_path):
        return font_manager.FontProperties(fname=font_path)
    else:
        print("[Warning] 微軟正黑體未找到，使用系統預設字型")
        return None

chinese_font = _load_chinese_font()
# ===========================================================

class ResultVisualizer:
    def __init__(self, df, rules):
        self.df = df.copy()
        self.rules = rules
        os.makedirs("charts", exist_ok=True)

    # --------------------------------------------------------------
    def plot_sentiment_pie_final(self):
        """Pie Chart: English"""
        counts = self.df['sentiment'].value_counts()
        if counts.empty: return
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#66b3ff', '#ff9999', '#ffcc99']
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
                startangle=90, textprops={'fontsize': 13})
        plt.title('Sentiment Distribution\n(Predicted by SVM Classifier)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig("charts/sentiment_pie_final.png", dpi=200)
        plt.close()
        print("[Success] Saved: charts/sentiment_pie_final.png")

    def plot_cluster_sentiment_with_topics(self):
        if 'cluster' not in self.df.columns:
            print("[Warning] No cluster column.")
            return
        data = self.df.groupby(['cluster', 'sentiment']).size().unstack(fill_value=0)
        vectorizer = TfidfVectorizer(token_pattern=r'\S+', max_features=3)
        cluster_keywords = {}
        for cid in sorted(self.df['cluster'].unique()):
            texts = self.df[self.df['cluster'] == cid]['tokens'].fillna('')
            text_str = ' '.join(texts)
            if text_str.strip():
                X = vectorizer.fit_transform([text_str])
                keywords = vectorizer.get_feature_names_out()
                cluster_keywords[cid] = ', '.join(keywords)
            else:
                cluster_keywords[cid] = 'Unknown'
        plt.figure(figsize=(11, 6))
        ax = data.plot(kind='bar', stacked=True,color=['#66b3ff', '#ff9999', '#ffcc99'],edgecolor='black', linewidth=0.5)
        #繪製標題
        plt.title('Cluster Sentiment Distribution\n(with Keywords)', pad=25)
        plt.xlabel('Cluster (Topic)')
        plt.ylabel('Count', fontsize=12)
        plt.legend(['Positive', 'Negative', 'Neutral'], title='Sentiment', loc='upper right')
        # 關鍵：xticks 標籤強制中文字型
        labels = [f"Cluster {i}\n({cluster_keywords[i]})" for i in data.index]
        ticks = ax.set_xticks(range(len(labels)))
        ticklabels = ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
        # 強制每個 tick label 使用中文字型
        if chinese_font:
            for label in ticklabels:
                label.set_fontproperties(chinese_font)        
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fontsize=9, fontweight='bold', color='white', fmt='%d')
        plt.tight_layout()
        plt.savefig("charts/cluster_sentiment_topics.png", dpi=200)
        plt.close()
        print("[Success] Saved: charts/cluster_sentiment_topics.png")
    # --------------------------------------------------------------
    
    def plot_rules_table(self):
        """Top 10 Rules: 中文欄位 + 強制中文字型"""
        if self.rules.empty:
            print("[Warning] No rules.")
            return
        top10 = self.rules.nlargest(10, 'support').copy()
        top10 = top10[['antecedents', 'consequents', 'support', 'confidence']]
        top10['antecedents'] = top10['antecedents'].apply(lambda x: ', '.join(list(x)))
        top10['consequents'] = top10['consequents'].apply(lambda x: ', '.join(list(x)))
        top10['support'] = top10['support'].round(6)
        top10['confidence'] = top10['confidence'].round(6)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.axis('tight'); ax.axis('off')

        # 強制修整表格文字
        table = ax.table(cellText=top10.values,
                         colLabels=['antecedents', 'consequents', 'Support', 'Confidence'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.2)

        # 強制每格文字用中文字型
        for i in range(len(top10) + 1):
            for j in range(4):
                cell = table.get_celld()[(i, j)]
                cell.get_text().set_fontproperties(chinese_font if chinese_font else font_manager.FontProperties())
        if chinese_font:
            plt.title('Top 10 關聯規則 (依支持度排序)', fontsize=16, fontweight='bold', pad=30, fontproperties=chinese_font)
        else:
            plt.title('Top 10 Association Rules (by Support)', fontsize=16, fontweight='bold', pad=30)
        plt.tight_layout()
        plt.savefig("charts/association_rules_table.png", dpi=200)
        plt.close()
        print("[Success] Saved: charts/association_rules_table.png")

    # --------------------------------------------------------------
    def show_rules_table(self):
        """Terminal: English"""
        if self.rules.empty:
            print("[Warning] No association rules.")
            return
        top10 = self.rules.nlargest(10, 'support').copy()
        top10 = top10[['antecedents', 'consequents', 'support', 'confidence']]
        top10['antecedents'] = top10['antecedents'].apply(lambda x: ', '.join(list(x)))
        top10['consequents'] = top10['consequents'].apply(lambda x: ', '.join(list(x)))
        print("\n=== Top 10 Association Rules (by Support) ===")
        print(top10.to_string(index=False, float_format='%.6f'))

    # --------------------------------------------------------------
    def plot_classification_comparison(self, clf_results):
        """SVM vs DT: English + Value Labels on Bars"""
        if not clf_results:
            print("[Warning] No classification results.")
            return
        models, accuracy, f1_macro = [], [], []
        for name, res in clf_results.items():
            models.append(name)
            accuracy.append(res['accuracy'])
            f1_macro.append(res['f1_macro'])
        x = np.arange(len(models))
        width = 0.35
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - width/2, accuracy, width, label='Accuracy', color='#66b3ff')
        bars2 = plt.bar(x + width/2, f1_macro, width, label='F1-Macro', color='#ff9999')
        plt.title('Classification Performance Comparison\n(SVM vs Decision Tree)', 
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(x, models)
        plt.ylim(0, 1)
        plt.legend()
        # 關鍵：加上數值標註
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig("charts/classification_comparison.png", dpi=200)
        plt.close()
        print("[Success] Saved: charts/classification_comparison.png")
    # --------------------------------------------------------------
    def plot_tfidf_table(self, tfidf_df):
        """TF-IDF Table: 中文關鍵詞 + 強制中文字型"""
        if tfidf_df.empty:
            print("[Warning] No TF-IDF data.")
            return

        df_plot = tfidf_df.sort_values('tfidf_score', ascending=False).copy()
        df_plot['tfidf_score'] = df_plot['tfidf_score'].round(6)

        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.axis('tight'); ax.axis('off')

        table = ax.table(cellText=df_plot.values,
                         colLabels=['關鍵詞', 'TF-IDF Score'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.2)

        # 強制每格文字用中文字型
        for i in range(len(df_plot) + 1):
            for j in range(2):
                cell = table.get_celld()[(i, j)]
                cell.get_text().set_fontproperties(chinese_font if chinese_font else font_manager.FontProperties())

        if chinese_font:
            plt.title('Top 15 TF-IDF 關鍵詞 (依分數排序)', fontsize=16, fontweight='bold', pad=35, fontproperties=chinese_font)
        else:
            plt.title('Top 15 TF-IDF Keywords (by Score)', fontsize=16, fontweight='bold', pad=35)

        plt.tight_layout()
        plt.savefig("charts/tfidf_keywords_table.png", dpi=200)
        plt.close()
        print("[Success] Saved: charts/tfidf_keywords_table.png")