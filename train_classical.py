import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

# Load the cleaned education policy dataset (100 policies)
full_df = pd.read_csv("education_policies.csv")
print(f"✅ Loaded dataset: {len(full_df)} policies from education_policies.csv")

def preprocess(df):
    df = df.copy()
    # Use all string columns for NLP
    text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith('str')]
    if not text_cols:
        raise ValueError("No text columns found in dataframe")
    df["text_for_nlp"] = df[text_cols].fillna('').agg(' '.join, axis=1).str.lower()
    return df

# Preprocess the full dataset
full_df = preprocess(full_df)
print("✅ Text preprocessing complete")

# Train TF-IDF on the full dataset for better accuracy
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=1)

vectorizer.fit(full_df["text_for_nlp"])
tfidf_matrix = vectorizer.transform(full_df["text_for_nlp"]).toarray()

# Save model + matrix
joblib.dump(vectorizer, MODEL_PATH)
joblib.dump({"tfidf_matrix": tfidf_matrix, "df": full_df}, MATRIX_PATH)

print(f"✅ Model trained and saved to {MODEL_PATH} and {MATRIX_PATH}")
print(f"Matrix shape: {tfidf_matrix.shape}")
print(f"Vocab size: {len(vectorizer.vocabulary_)}")