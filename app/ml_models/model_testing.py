# -*- coding: utf-8 -*-
"""model_testing.ipynb"""
#!pip install gdown nltk wordcloud xgboost lightgbm scikit-learn matplotlib seaborn joblib gensim
# Imports
import pandas as pd
import numpy as np
import re
import string
import gdown
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold
# from google.colab import files  # Commented for local use
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api
from gensim.models import KeyedVectors
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')


# Resume Classification Class
class ResumeClassifier:
    def __init__(self, df):
        self.df = df.copy()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',
                                                max_features=3000, ngram_range=(1, 2))
        self.word2vec_model = None
        self.models = {}

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"\S+@\S+", '', text)
        text = re.sub(r"@\w+|#", '', text)
        text = re.sub(r"[^a-zA-Z\s]", ' ', text)
        text = re.sub(r"\s+", ' ', text).strip()
        return text

    def preprocess(self):
        self.df['cleaned_text'] = self.df['Resume_str'].apply(self.clean_text)
        self.df['tokens'] = self.df['cleaned_text'].apply(lambda x: x.split())
        self.df['encoded_label'] = self.label_encoder.fit_transform(self.df['category'])

        # Optional: Save label names to file
        with open("category_list.txt", "w") as f:
            for cat in self.label_encoder.classes_:
                f.write(cat + "\n")

    def load_word2vec(self):
        try:
            self.word2vec_model = KeyedVectors.load("word2vec_model.kv")
            print("Word2Vec loaded from disk.")
        except:
            print("Downloading Word2Vec model...")
            self.word2vec_model = api.load("word2vec-google-news-300")
            self.word2vec_model.save("word2vec_model.kv")
            print("Word2Vec downloaded and saved.")

    def get_word2vec_features(self, tokens, tfidf_idf=None):
        vec = np.zeros(300)
        total_weight = 0
        for word in tokens:
            if word in self.word2vec_model:
                weight = tfidf_idf[word] if tfidf_idf and word in tfidf_idf else 1
                vec += self.word2vec_model[word] * weight
                total_weight += weight
        return vec / total_weight if total_weight > 0 else vec

    def build_features(self, mode="hybrid"):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['cleaned_text'])
        idf_scores = dict(zip(self.tfidf_vectorizer.get_feature_names_out(),
                              self.tfidf_vectorizer.idf_))

        if mode == "tfidf":
            return tfidf_matrix

        if mode == "word2vec":
            vectors = np.vstack(self.df['tokens'].apply(lambda x: self.get_word2vec_features(x)))
            return csr_matrix(vectors)

        if mode == "hybrid":
            vectors = np.vstack(self.df['tokens'].apply(lambda x: self.get_word2vec_features(x, idf_scores)))
            return hstack([tfidf_matrix, csr_matrix(vectors)])

    def train_and_evaluate(self, X, y):
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        xgbm = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        lgbm = lgb.LGBMClassifier(random_state=42)

        model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgbm), ('lgbm', lgbm)],
            voting='soft',
            weights=[1, 2, 3]
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            scores.append(accuracy_score(y[test_idx], preds))
        return model, np.mean(scores)

    def auto_select_best_model(self):
        y = self.df['encoded_label'].values
        self.load_word2vec()

        print("Evaluating TF-IDF Only...")
        X_tfidf = self.build_features("tfidf")
        model_tfidf, score_tfidf = self.train_and_evaluate(X_tfidf, y)
        print(f"TF-IDF Accuracy: {score_tfidf:.4f}")

        print("Evaluating Word2Vec Only...")
        X_w2v = self.build_features("word2vec")
        model_w2v, score_w2v = self.train_and_evaluate(X_w2v, y)
        print(f"Word2Vec Accuracy: {score_w2v:.4f}")

        print("Evaluating Hybrid (TF-IDF + Word2Vec)...")
        X_hybrid = self.build_features("hybrid")
        model_hybrid, score_hybrid = self.train_and_evaluate(X_hybrid, y)
        print(f"Hybrid Accuracy: {score_hybrid:.4f}")

        best_model, best_X, best_score, best_name = max(
            [(model_tfidf, X_tfidf, score_tfidf, "TF-IDF"),
             (model_w2v, X_w2v, score_w2v, "Word2Vec"),
             (model_hybrid, X_hybrid, score_hybrid, "Hybrid")],
            key=lambda x: x[2]
        )

        print(f"Best model selected: {best_name} (Accuracy: {best_score:.4f})")

        # ✅ Save the name of best model type
        with open("best_mode.txt", "w") as f:
            f.write(best_name.lower())  # tfidf / word2vec / hybrid

        return best_model, best_X, y

    def final_evaluation(self, model, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print("Final Evaluation on Hold-Out Set:")
        print(classification_report(y_test, preds, target_names=self.label_encoder.classes_))

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        return model

    def save_model(self, model):
        joblib.dump(model, 'Resume_Ensemble_Model.pkl')
        joblib.dump(self.label_encoder, 'Resume_LabelEncoder.pkl')
        joblib.dump(self.tfidf_vectorizer, 'Resume_Tfidf_Vectorizer.pkl')
        print("Model and preprocessing artifacts saved.")

    def get_related_categories(self, label, topn=3, threshold=0.8):
        if not self.word2vec_model:
            self.load_word2vec()

        label_words = label.lower().split()
        related = {}

        for cat in self.label_encoder.classes_:
            if cat == label:
                continue
            cat_words = cat.lower().split()
            sims = []
            for lw in label_words:
                for cw in cat_words:
                    if lw in self.word2vec_model and cw in self.word2vec_model:
                        sims.append(self.word2vec_model.similarity(lw, cw))
            if sims:
                avg_sim = np.mean(sims)
                if avg_sim >= threshold:
                    related[cat] = round(avg_sim, 3)

        return dict(sorted(related.items(), key=lambda x: x[1], reverse=True)[:topn])


# Inference Function (for single testing)
def predict_resume_category(text):
    model = joblib.load('Resume_Ensemble_Model.pkl')
    label_encoder = joblib.load('Resume_LabelEncoder.pkl')
    vectorizer = joblib.load('Resume_Tfidf_Vectorizer.pkl')

    cleaned = ResumeClassifier.clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]


# Main Execution
if __name__ == "__main__":
    file_id = '1lX3TbYHy1ddyfvFwQsF9HhHcjgjswxaN'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', 'Big_Resumes_Dataset.csv', quiet=False)
    df = pd.read_csv('Big_Resumes_Dataset.csv')
    df['category'].value_counts().sort_index().plot(kind='bar', figsize=(12, 6))
    plt.show()

    clf = ResumeClassifier(df)
    clf.preprocess()
    model, X_final, y_final = clf.auto_select_best_model()
    final_model = clf.final_evaluation(model, X_final, y_final)
    clf.save_model(final_model)
