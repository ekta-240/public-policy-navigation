# 🔎 Public Policy Navigation System using AI

## Overview
This project is a modern, production-ready AI-powered web application for searching, analyzing, and visualizing education policies. It leverages both classical (TF-IDF) and quantum-inspired (PennyLane) NLP models to help users find the most relevant policies using natural language queries. The app features interactive analytics, a clean UI, and is designed for easy deployment and extension.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ekta-240/public-policy-navigation.git
cd public-policy-navigation-using-ai-

# 2. (Optional) Create and activate a virtual environment
# python -m venv venv
# .\venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn jinja2 joblib pandas scikit-learn numpy pennylane

# 4. Train models if needed
python train_classical.py
python quantum_nlp_train.py

# 5. Run the application
uvicorn app:app --reload --host 0.0.0.0 --port 5010

# 6. Open in browser
# Visit: http://localhost:5010
```

## Project Structure
```
public-policy-navigation-using-ai-/
├── app.py                        # FastAPI app: main backend, search logic, API, UI rendering
├── train_classical.py            # Script to train classical TF-IDF model
├── quantum_nlp_train.py          # Script to train quantum (PennyLane) model
├── education_policies.csv        # Main dataset (500 policies)
├── train_policies.csv            # Training split (classical)
├── test_policies.csv             # Test split (classical)
├── trainq1_policies.csv          # Training split (quantum)
├── testq1_policies.csv           # Test split (quantum)
├── policy_vectorizer.pkl         # Trained TF-IDF vectorizer (classical)
├── policy_tfidf_matrix.pkl       # TF-IDF matrix (classical)
├── policyq1_vectorizer.pkl       # Trained TF-IDF vectorizer (quantum)
├── policyq1_tfidf_matrix_quantum.pkl # Quantum model matrix
├── infosys_edu_quantum.ipynb     # Jupyter notebook: quantum NLP experiments
├── infosys_nlp.ipynb             # Jupyter notebook: classical NLP experiments
├── static/
│   └── chart.js                  # Chart.js for frontend analytics
├── templates/
│   └── index.html                # Jinja2 HTML template for UI
├── __pycache__/                  # Python cache
└── README.md                     # Project documentation
```

## ✨ Features
- **Classical & Quantum Search:** Find relevant education policies using traditional TF-IDF or quantum-inspired NLP (PennyLane simulation).
- **Interactive Analytics:** Visualize policy distributions by year, region, and relevance using Chart.js (for quantum search).
- **Modern UI:** Responsive, clean interface with Jinja2 templates and custom CSS.
- **Jupyter Notebooks:** Explore and experiment with both classical and quantum NLP approaches.
- **Production-Ready:** Only essential code, models, and data included.

   ---


## 🛠️ Technology Stack
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML5, CSS3, Jinja2, Chart.js
- **ML/NLP:** scikit-learn (TF-IDF), PennyLane (quantum simulation)
- **Data Processing:** pandas, numpy

   ---


## 📊 Usage
1. **Enter a Query:** Type your search (e.g., "girls education policy", "teacher training", "digital access in rural areas").
2. **Choose Search Type:** Select "Classical" or "Quantum" search.
3. **View Results:** Top 3 most relevant policies are shown with:
   - Title, region, year, score, and summary
   - Interactive charts (for quantum search only)

   ---

## 📦 Files & Their Purpose
- `app.py`: Main FastAPI server, search logic, API, UI rendering
- `train_classical.py`: Trains and saves classical TF-IDF model
- `quantum_nlp_train.py`: Trains and saves quantum (PennyLane) model
- `education_policies.csv`: Main dataset (500 policies)
- `train_policies.csv`, `test_policies.csv`: Classical model train/test splits
- `trainq1_policies.csv`, `testq1_policies.csv`: Quantum model train/test splits
- `policy_vectorizer.pkl`, `policy_tfidf_matrix.pkl`: Classical model artifacts
- `policyq1_vectorizer.pkl`, `policyq1_tfidf_matrix_quantum.pkl`: Quantum model artifacts
- `infosys_edu_quantum.ipynb`: Jupyter notebook for quantum NLP experiments
- `infosys_nlp.ipynb`: Jupyter notebook for classical NLP experiments
- `static/chart.js`: Chart.js for frontend analytics
- `templates/index.html`: Main HTML template for the UI
- `__pycache__/`: Python bytecode cache

---

**Repository:** [public-policy-navigation](https://github.com/ekta-240/public-policy-navigation)
- Built as part of the Infosys Springboard learning program
=======
  
<img width="1920" height="1080" alt="Screenshot (2976)" src="https://github.com/user-attachments/assets/1fbde44b-1a17-4dff-8a56-5b09a5c72ee1" />
<img width="1920" height="1080" alt="Screenshot (2977)" src="https://github.com/user-attachments/assets/4151871a-410d-432a-94f3-c96983e715da" />
<img width="1920" height="1080" alt="Screenshot (2978)" src="https://github.com/user-attachments/assets/66d92752-3287-4a10-84db-b42274aa5171" />
<img width="1920" height="1080" alt="Screenshot (2979)" src="https://github.com/user-attachments/assets/f2abb77b-4703-4eba-8b49-15cdc8bd7ac3" />
