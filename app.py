"""
AI-Powered Policy Navigation System
Integrates both Classical TF-IDF and Quantum-Enhanced NLP search
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys
import joblib

# Import quantum search with fallback
try:
    from quantum_nlp_train import quantum_search, check_model_status
    QUANTUM_AVAILABLE = True
    print("✅ Quantum NLP module loaded successfully")
except ImportError as e:
    QUANTUM_AVAILABLE = False
    print(f"⚠️  Quantum NLP module not available: {e}")
    print("   Quantum search will fall back to classical search")


app = FastAPI(title="AI-Powered Policy Navigation System")

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    print("⚠️  Static directory not found")

# Setup templates
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")
else:
    print("❌ Templates directory not found!")
    sys.exit(1)

# Global variables for models
vectorizer = None
tfidf_matrix = None
policies_df = None

def load_classical_models():
    """Load classical TF-IDF models"""
    global vectorizer, tfidf_matrix, policies_df
    
    try:
        print("\n" + "=" * 60)
        print("LOADING CLASSICAL MODELS")
        print("=" * 60)
        
        # Load vectorizer
        if os.path.exists('policy_vectorizer.pkl'):
            vectorizer = joblib.load('policy_vectorizer.pkl')
            print("✅ Loaded policy_vectorizer.pkl")
        else:
            print("❌ policy_vectorizer.pkl not found")
            return False
        
        # Load TF-IDF matrix
        if os.path.exists('policy_tfidf_matrix.pkl'):
            data = joblib.load('policy_tfidf_matrix.pkl')
            tfidf_matrix = data["tfidf_matrix"]
            policies_df = data["df"]
            print(f"✅ Loaded policy_tfidf_matrix.pkl (shape: {tfidf_matrix.shape})")
            print(f"✅ Loaded policies df ({len(policies_df)} policies)")
        else:
            print("❌ policy_tfidf_matrix.pkl not found")
            return False
        
        print("=" * 60)
        print("✅ ALL CLASSICAL MODELS LOADED SUCCESSFULLY")
        print("=" * 60 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Error loading classical models: {e}")
        import traceback
        traceback.print_exc()
        return False

def search_policies(query, top_k=3):
    """
    Search for relevant policies using Classical TF-IDF vectorization
    
    Args:
        query (str): User's search query
        top_k (int): Number of top results to return
    
    Returns:
        list: List of dictionaries containing policy information and relevance scores
    """
    global vectorizer, tfidf_matrix, policies_df
    
    if vectorizer is None or tfidf_matrix is None or policies_df is None:
        raise ValueError("Classical models not loaded. Please ensure model files exist.")
    
    # Transform the query using the pre-trained vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all policies
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Prepare results
    results = []
    for idx in top_indices:
        policy = policies_df.iloc[idx]
        
        # Handle different column name variations
        title = policy.get('title', 'Untitled Policy')
        # Try to get summary, else generate from title and goals
        description = policy.get('summary', '')
        if not description:
            # Try to generate summary from title and goals
            goals = policy.get('goals', '')
            snippet = str(title) + '. ' + str(goals)
            description = snippet[:250] + ('...' if len(snippet) > 250 else '')
        category = policy.get('region', 'General')
        
        results.append({
            'title': str(title),
            'summary': str(description),
            'region': str(category),
            'score': float(similarities[idx]),
            'policy_id': int(idx),
            'year': policy.get('year', ''),
            'status': 'Unknown'
        })
    
    return results

def quantum_search_wrapper(query, top_k=3):
    """
    Wrapper for quantum search with fallback to classical
    
    Args:
        query (str): User's search query
        top_k (int): Number of top results to return
    
    Returns:
        tuple: (results list, search_method string)
    """
    if QUANTUM_AVAILABLE:
        try:
            # Check if quantum models are trained
            status = check_model_status()
            
            if True:  # Force quantum
                print(f"Using Quantum Search")
                results = quantum_search(query, top_k=top_k)
                return results, "Quantum-Enhanced"
            else:
                print("Quantum models not trained. Falling back to classical search.")
                results = search_policies(query, top_k)
                return results, "Classical TF-IDF (Quantum models not trained)"
                
        except Exception as e:
            print(f"Quantum search failed: {e}. Falling back to classical search.")
            results = search_policies(query, top_k)
            return results, "Classical TF-IDF (Quantum search failed)"
    else:
        print("Quantum module not available. Using classical search.")
        results = search_policies(query, top_k)
        return results, "Classical TF-IDF (Quantum module unavailable)"

@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    print("\n" + "=" * 60)
    print("🚀 STARTING AI-POWERED POLICY NAVIGATION SYSTEM")
    print("=" * 60)
    
    # Load classical models
    classical_loaded = load_classical_models()
    
    if not classical_loaded:
        print("\n❌ CRITICAL: Classical models failed to load!")
        print("Please ensure the following files exist:")
        print("  - policy_vectorizer.pkl")
        print("  - policy_tfidf_matrix.pkl")
        print("  - education_policies100_cleaned.csv")
        print("\nApplication will continue but search will not work.")
    
    # Check quantum availability
    if QUANTUM_AVAILABLE:
        status = check_model_status()
        if status["ready"]:
            print(f"\n✅ Quantum models ready: {status['n_policies']} policies, {status['n_qubits']} qubits")
        else:
            print("\n⚠️  Quantum models not trained yet.")
            print("Run: python quantum_nlp_train.py")
    
    print("\n" + "=" * 60)
    print("✅ APPLICATION STARTUP COMPLETE")
    print("=" * 60 + "\n")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "quantum_available": QUANTUM_AVAILABLE
    })

@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request, 
    query: str = Form(...), 
    search_type: str = Form("classical")
):
    """
    Handle search requests
    
    Args:
        request: FastAPI request object
        query: User's search query
        search_type: Type of search (classical or quantum)
    
    Returns:
        HTMLResponse: Rendered template with search results
    """
    # Validate query
    if not query or query.strip() == "":
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Please enter a search query",
                "quantum_available": QUANTUM_AVAILABLE
            }
        )
    
    try:
        print(f"\n{'=' * 60}")
        print(f"Search Request: '{query}' (type: {search_type})")
        print(f"{'=' * 60}")
        
        # Perform search based on type
        if search_type == "quantum":
            results, search_method = quantum_search_wrapper(query, top_k=3)
        else:
            results = search_policies(query, top_k=3)
            search_method = "Classical TF-IDF"
        
        # Add search method to each result
        results = [
            {**result, 'search_method': search_method} 
            for result in results
        ]
        
        # Generate graphs for quantum search
        graphs = None
        region_counts = None
        year_counts = None
        if policies_df is not None:
            region_counts = policies_df['region'].value_counts().to_dict()
            year_counts = policies_df['year'].value_counts().sort_index().to_dict()
        if search_type == "quantum" and results:
            scores = [float(r.get('score', 0)) for r in results]
            # For quantum, still send filtered years/regions for legacy code, but also send full dataset
            years = {}
            regions = {}
            for r in results:
                year = str(r.get('year', 'Unknown'))
                region = str(r.get('region', 'Unknown'))
                years[year] = years.get(year, 0) + 1
                regions[region] = regions.get(region, 0) + 1
            graphs = {
                'scores': scores,
                'years': year_counts,  # use full dataset
                'regions': region_counts  # use full dataset
            }
        
        print(f"Found {len(results)} results using {search_method}")
        print(f"{'=' * 60}\n")
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "query": query,
                "results": results,
                "search_type": search_type,
                "graphs": graphs,
                "region_counts": region_counts,
                "year_counts": year_counts,
                "quantum_available": QUANTUM_AVAILABLE
            }
        )
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Search failed: {str(e)}",
                "query": query,
                "quantum_available": QUANTUM_AVAILABLE
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    classical_ready = (vectorizer is not None and 
                      tfidf_matrix is not None and 
                      policies_df is not None)
    
    quantum_status = {"available": False, "ready": False}
    if QUANTUM_AVAILABLE:
        quantum_status = check_model_status()
        quantum_status["available"] = True
    
    return {
        "status": "healthy" if classical_ready else "degraded",
        "classical": {
            "ready": classical_ready,
            "policies_loaded": len(policies_df) if policies_df is not None else 0,
            "vectorizer_loaded": vectorizer is not None,
            "tfidf_matrix_shape": tfidf_matrix.shape if tfidf_matrix is not None else None
        },
        "quantum": quantum_status
    }

@app.get("/api/search")
async def api_search(query: str, search_type: str = "classical", top_k: int = 3):
    """
    API endpoint for programmatic search
    
    Args:
        query: Search query
        search_type: 'classical' or 'quantum'
        top_k: Number of results to return
    
    Returns:
        JSON response with search results
    """
    try:
        if search_type == "quantum":
            results, search_method = quantum_search_wrapper(query, top_k=top_k)
        else:
            results = search_policies(query, top_k=top_k)
            search_method = "Classical TF-IDF"
        
        return {
            "results": results,
            "search_method": search_method,
            "query": query,
            "top_k": top_k
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "query": query
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)