from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class TextMiner:
    def __init__(self, df):
        self.df = df.copy()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.X = self.vectorizer.fit_transform(df['tokens'])
        self.y = df['sentiment']

        print("\n[TextMiner Init] === Input Overview ===")
        print(f"TF-IDF 矩陣 self.X 形狀: {self.X.shape}")
        print(f"情緒標籤 self.y 分佈:\n{self.y.value_counts()}")
        print(f"前 10 個 TF-IDF 特徵詞:\n{self.vectorizer.get_feature_names_out()[:10]}")
        print("======================================\n")

    # 分類模型
    def classify(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # --- 向量化前的部分文字與標籤 ---
        print("\n[Sample Raw Text & Labels] (前5筆)")
        for text, label in zip(self.df['tokens'].head(5), self.y.head(5)):
            print(f"Text: {text[:50]}... | Label: {label}")

        # --- 顯示部分 TF-IDF 矩陣（前5筆、全部特徵列） ---
        print("\n[TF-IDF Sample] (前5筆)")
        tfidf_df_sample = pd.DataFrame(
            self.X[:5, :].toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        print(tfidf_df_sample)

        results = {}

        # --- SVM ---
        svm = SVC(kernel='linear', random_state=42)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
        results['SVM'] = {
            'report': report_svm,
            'accuracy': report_svm['accuracy'],
            'f1_macro': report_svm['macro avg']['f1-score']
        }
        print(f"SVM Accuracy: {results['SVM']['accuracy']:.3f}")

        # --- 決策樹 ---
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
        results['Decision Tree'] = {
            'report': report_dt,
            'accuracy': report_dt['accuracy'],
            'f1_macro': report_dt['macro avg']['f1-score']
        }
        print(f"DT Accuracy: {results['Decision Tree']['accuracy']:.3f}")

        # --- 部分測試預測結果 ---
        print("\n[Sample Predictions] (前5筆)")
        for text, pred in zip(self.df['tokens'].head(5), y_pred_svm[:5]):
            print(f"Text: {text[:50]}... | Predicted: {pred}")

        return results

    # 分群模型
    def cluster(self, n_clusters=3):
        print(f"\n[Clustering] Running KMeans (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X)

        score = silhouette_score(self.X, labels)
        inertia = kmeans.inertia_
        print(f"K-means 輪廓係數: {score:.3f} | 慣性: {inertia:.1f}")

        self.df['cluster'] = labels

        # --- 顯示部分群集結果 ---
        print("\n[Sample Cluster Assignments] (前5筆)")
        print(self.df[['tokens','cluster']].head(5))

        return {
            'silhouette_score': score,
            'inertia': inertia,
            'n_clusters': n_clusters,
            'labels': labels
        }

    # Apriori
    def association_rules(self, min_support=0.02, min_confidence=0.5):
        # 原始 TF-IDF 矩陣（前5筆、全部特徵列）
        print("\n[TF-IDF Sample] (前5筆)")
        tfidf_df_sample = pd.DataFrame(
            self.X[:5, :].toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        print(tfidf_df_sample)

        # 布林化
        freq_matrix = pd.DataFrame(
            self.X.toarray() > 0,
            columns=self.vectorizer.get_feature_names_out()
        )
        print("\n[Boolean TF-IDF Sample] (前5筆)")
        print(freq_matrix.head(5))

        # Apriori 挖掘
        frequent_itemsets = apriori(freq_matrix, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if not rules.empty:
            print(f"\n[Apriori] 發現 {len(rules)} 條規則")
            print("\n[Sample Rules] (前5筆)")
            print(rules[['antecedents','consequents','support','confidence']].head(5))
        else:
            print("Apriori: 無符合條件的規則")

        return rules
