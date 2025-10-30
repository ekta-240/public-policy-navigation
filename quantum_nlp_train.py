"""
PennyLane Quantum NLP for Policy Search
Integrated from infosys_edu_quantum.ipynb
This module provides quantum-enhanced search functionality for education policies
using PennyLane quantum circuits and TF-IDF vectorization.
"""

import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "policyq1_vectorizer.pkl"
MATRIX_PATH = "policyq1_tfidf_matrix_quantum.pkl"
TRAIN_DATA_PATH = "education_policies100_cleaned.csv"
FULL_DATA_PATH = "education_policies100_cleaned.csv"

# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess(df):
    """
    Preprocess the dataframe by combining text columns
    
    Args:
        df (DataFrame): Input dataframe with policy data
    
    Returns:
        DataFrame: Preprocessed dataframe with combined text column
    """
    df = df.copy()
    
    # Try to find available text columns (flexible column names)
    text_cols = ["title", "Title", "summary", "Summary", "goals", "Goals", "description", "Description"]
    available = [c for c in text_cols if c in df.columns]
    
    if not available:
        raise ValueError("No text columns found in dataframe")
    
    # Combine available text columns
    df["text_for_nlp"] = df[available].fillna('').agg(' '.join, axis=1)
    
    # Add stakeholders if available
    if "stakeholders" in df.columns:
        df["text_for_nlp"] = df["text_for_nlp"] + ". Stakeholders: " + df["stakeholders"].astype(str)
    elif "Stakeholders" in df.columns:
        df["text_for_nlp"] = df["text_for_nlp"] + ". Stakeholders: " + df["Stakeholders"].astype(str)
    
    # Convert to lowercase for consistency
    df["text_for_nlp"] = df["text_for_nlp"].str.lower()
    
    return df

# ============================================================================
# TRAINING
# ============================================================================
def train_quantum_model(max_features=16, max_policies=50):
    """
    Train the quantum NLP model with TF-IDF vectorization
    
    Args:
        max_features (int): Maximum number of TF-IDF features (must match qubit count)
        max_policies (int): Maximum number of policies to process (for performance)
    
    Returns:
        bool: True if training successful, False otherwise
    """
    print("=" * 60)
    print("QUANTUM NLP MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Load datasets
        print("\n1. Loading datasets...")
        
        # Try different file names
        if os.path.exists(TRAIN_DATA_PATH):
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            print(f"   ✅ Loaded training data: {len(train_df)} policies from {TRAIN_DATA_PATH}")
        else:
            print(f"   ⚠️  Training data not found, using full dataset for training")
            train_df = pd.read_csv(FULL_DATA_PATH)
        full_df = pd.read_csv(FULL_DATA_PATH)
        print(f"   ✅ Loaded full dataset: {len(full_df)} policies from {FULL_DATA_PATH}")
        
        # Use all policies in the dataset, do not limit by max_policies
        print(f"   ✅ Using all {len(full_df)} policies for training")
        
        # Preprocess
        print("\n2. Preprocessing text data...")
        train_df = preprocess(train_df)
        full_df = preprocess(full_df)
        print("   ✅ Text preprocessing complete")
        
        # Vectorize with TF-IDF
        print(f"\n3. Vectorizing with TF-IDF (max_features={max_features})...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1
        )
        vectorizer.fit(train_df["text_for_nlp"])
        tfidf_matrix = vectorizer.transform(full_df["text_for_nlp"]).toarray()
        print(f"   ✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"   ✅ Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Save model
        print("\n4. Saving model artifacts...")
        joblib.dump(vectorizer, MODEL_PATH)
        joblib.dump({
            "tfidf_matrix": tfidf_matrix,
            "df": full_df,
            "n_qubits": max_features
        }, MATRIX_PATH)
        print(f"   ✅ Model saved to {MODEL_PATH}")
        print(f"   ✅ Matrix saved to {MATRIX_PATH}")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# QUANTUM CIRCUIT DEFINITION
# ============================================================================
def create_feature_map(n_qubits):
    """
    Create a quantum feature map for encoding classical data
    
    Args:
        n_qubits (int): Number of qubits in the circuit
    
    Returns:
        function: Feature map function
    """
    def feature_map(x):
        """Apply rotation gates and entanglement"""
        # Apply RY rotations based on input features
        for i in range(n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
        
        # Apply entanglement with CNOT gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Additional layer for deeper encoding
        for i in range(n_qubits):
            qml.RZ(x[i] * np.pi / 2, wires=i)
    
    return feature_map

# ============================================================================
# QUANTUM SEARCH
# ============================================================================
def quantum_search(query, top_k=3):
    """
    Perform quantum-enhanced policy search using PennyLane
    
    Args:
        query (str): Search query text
        top_k (int): Number of top results to return
    
    Returns:
        list: List of dictionaries containing search results with scores
    """
    try:
        print(f"\nQuantum Search: '{query}'")
        
        # Load model artifacts
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MATRIX_PATH):
            print("Model files not found. Please train the model first.")
            return []
        
        vectorizer = joblib.load(MODEL_PATH)
        data = joblib.load(MATRIX_PATH)
        tfidf_matrix = data["tfidf_matrix"]
        df = data["df"]
        n_qubits = data.get("n_qubits", tfidf_matrix.shape[1])
        
        print(f"   Loaded model (n_qubits={n_qubits}, policies={len(df)})")
        
        # Initialize quantum device
        dev = qml.device("default.qubit", wires=n_qubits)
        feature_map = create_feature_map(n_qubits)
        
        # Define quantum circuit
        @qml.qnode(dev)
        def get_quantum_state(x):
            """Get quantum state for input vector"""
            feature_map(x)
            return qml.state()
        
        # Transform query to TF-IDF vector
        query_vec = vectorizer.transform([query.lower()]).toarray()
        
        # Get quantum state for query
        query_state = get_quantum_state(query_vec[0])
        
        # Calculate quantum fidelity with all documents
        print("   Computing quantum fidelities...")
        similarities = []
        for i in range(len(tfidf_matrix)):
            doc_state = get_quantum_state(tfidf_matrix[i])
            fidelity = np.abs(np.dot(np.conj(query_state), doc_state))**2
            similarities.append(float(fidelity))
        
        similarities = np.array(similarities)
        
        # Get top-k results
        top_indices = similarities.argsort()[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            
            # Handle different column name variations
            title = row.get("title", row.get("Title", "Untitled Policy"))
            description = ""
            
            if "summary" in row:
                description = str(row["summary"])
            elif "Summary" in row:
                description = str(row["Summary"])
            elif "description" in row:
                description = str(row["description"])
            elif "Description" in row:
                description = str(row["Description"])
            
            if "goals" in row and pd.notna(row["goals"]):
                description += " " + str(row["goals"])
            elif "Goals" in row and pd.notna(row["Goals"]):
                description += " " + str(row["Goals"])
            
            result = {
                "title": title,
                "summary": description.strip(),
                "region": row.get("region", row.get("Region", "General")),
                "score": round(float(similarities[idx]), 4),
                "policy_id": row.get("policy_id", row.get("Policy_ID", idx)),
                "year": row.get("year", row.get("Year", "")),
                "status": "Unknown"
            }
            
            # Add optional fields if available
            if "year" in row:
                result["year"] = row["year"]
            elif "Year" in row:
                result["year"] = row["Year"]
            
            results.append(result)
        
        print(f"   Found {len(results)} results")
        return results
        
    except Exception as e:
        print(f"Error in quantum_search: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def check_model_status():
    """
    Check if quantum model is trained and ready
    
    Returns:
        dict: Status information
    """
    status = {
        "model_exists": os.path.exists(MODEL_PATH),
        "matrix_exists": os.path.exists(MATRIX_PATH),
        "ready": False
    }
    
    if status["model_exists"] and status["matrix_exists"]:
        try:
            data = joblib.load(MATRIX_PATH)
            status["ready"] = True
            status["n_policies"] = len(data["df"])
            status["n_qubits"] = data.get("n_qubits", data["tfidf_matrix"].shape[1])
        except Exception as e:
            status["error"] = str(e)
    
    return status

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("QUANTUM NLP POLICY SEARCH")
    print("=" * 60)
    
    # Always retrain and overwrite model files to use latest data
    print("\nTraining new quantum model with latest data...")
    success = train_quantum_model(max_features=4, max_policies=50)
    if not success:
        print("\nTraining failed. Exiting.")
        sys.exit(1)
    
    # Test quantum search
    print("\n" + "=" * 60)
    print("TESTING QUANTUM SEARCH")
    print("=" * 60)
    
    test_queries = [
        "teacher training and professional development",
        "student financial aid and scholarships",
        "digital learning and technology in education"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = quantum_search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Region: {result['region']}")
                print(f"   Relevance: {result['score']:.4f}")
                print(f"   Description: {result['summary'][:100]}...")
        else:
            print("   No results found")