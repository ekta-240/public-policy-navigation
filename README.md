# 🔎 Public Policy Navigation System using AI

   ## Overview
   This project is a modern, production-ready AI-powered web application for searching, analyzing, and visualizing education policies. It leverages both classical (TF-IDF) and quantum-inspired (PennyLane) NLP models to help users find the most relevant policies using natural language queries. The app features interactive analytics, a clean UI, and is designed for easy deployment and extension.

## 🚀 Quick Start

   1. **Clone the repository**
      ```bash
      git clone https://github.com/ekta-240/public-policy-navigation-using-ai.git
      cd public-policy-navigation-using-ai
      ```
   2. **Install dependencies**
      ```bash
      pip install -r requirements.txt
      pip install pennylane
      ```
   3. **Train models (if needed)**
      ```bash
      python train_classical.py
      python quantum_nlp_train.py
      ```
   4. **Run the application**
      ```bash
      uvicorn app:app --reload --port 8000
      ```
   5. **Open in browser**
      - Visit: [http://localhost:8000](http://localhost:8000)
```
```
 ## ✨ Features
   - **Classical & Quantum Search:** Find relevant education policies using traditional TF-IDF or quantum-inspired NLP (PennyLane simulation).
   - **Interactive Analytics:** Visualize policy distributions by year, region, and relevance using Chart.js.
   - **Modern UI:** Responsive, clean interface with Jinja2 templates and custom CSS.
   - **Production-Ready:** All unnecessary files removed; only essential code, models, and data included.

  
```
```
   ## 🛠️ Technology Stack
   - **Backend:** FastAPI, Uvicorn
   - **Frontend:** HTML5, CSS3, Jinja2, Chart.js
   - **ML/NLP:** scikit-learn (TF-IDF), PennyLane (quantum simulation)
   - **Data Processing:** pandas, numpy
```
```
## Project Structure
```
public-policy-navigation-using-ai/
├── app.py                      # FastAPI application (main server)
├── train_classical.py          # Script to train classical TF-IDF model
├── quantum_nlp_train.py        # Script to train quantum-inspired model
├── education_policies100_cleaned.csv # Main dataset (production)
├── policy_vectorizer.pkl       # Trained TF-IDF vectorizer (classical)
├── policy_tfidf_matrix.pkl     # Pre-computed TF-IDF matrix (classical)
├── policyq1_vectorizer.pkl     # Trained TF-IDF vectorizer (quantum)
├── policyq1_tfidf_matrix_quantum.pkl # Quantum model matrix
├── static/
│   └── chart.js                # Chart.js for frontend analytics
├── templates/
│   └── index.html              # Jinja2 HTML template for UI
└── README.md                   # Project documentation
```

   ## 📦 Files & Their Purpose
   - `app.py`: Main web server, handles search, loads models, serves UI.
   - `train_classical.py`: Trains and saves classical TF-IDF model.
   - `quantum_nlp_train.py`: Trains and saves quantum-inspired model (PennyLane simulation).
   - `education_policies100_cleaned.csv`: Main dataset for production use.
   - `policy_vectorizer.pkl`, `policy_tfidf_matrix.pkl`: Classical model artifacts.
   - `policyq1_vectorizer.pkl`, `policyq1_tfidf_matrix_quantum.pkl`: Quantum model artifacts.
   - `static/chart.js`: Chart.js library for frontend analytics.
   - `templates/index.html`: Main HTML template for the UI.
```
```

### Example Queries
- "early childhood education in Karnataka"
- "STEM programs for girls"
- "rural school infrastructure 2023"
- "teacher recruitment Maharashtra"
- "digital literacy initiatives"

## Dataset Information

The synthetic education policy dataset includes:
- **500 policies** covering various education sectors
- **Sectors**: Primary, Secondary, Higher Education, Vocational, Early Childhood
- **Regions**: 10 Indian states (Karnataka, Maharashtra, Tamil Nadu, etc.)
- **Years**: 2015-2025
- **Fields**: 
  - Policy ID, Title, Sector, Region, Year
  - Target Group, Status, Funding
  - Stakeholders, Impact Score
  - Summary, Goals, Full Text

## Model Details

- **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 5000 max features, bigrams (1-2 word combinations)
- **Similarity Metric**: Cosine Similarity
- **Training Data**: 400 policies
- **Test Data**: 100 policies

## License
  MIT License

   ---

   ## 📊 Usage
   1. **Enter a Query:** Type your search (e.g., "girls education policy", "teacher training", "digital access in rural areas").
   2. **Choose Search Type:** Select "Classical" or "Quantum" search.
   3. **View Results:** Top 3 most relevant policies are shown with:
      - Title, region, year, score, and summary
      - Highlighted keywords (e.g., "girls", "female", "women")
      - Interactive charts (for quantum search)

   ---

   ---

   **Repository:** [public-policy-navigation-using-ai](https://github.com/ekta-240/public-policy-navigation)
   **Status:** Production-ready, cleaned for GitHub
   **Last Updated:** October 2025
- Built as part of the Infosys Springboard learning program

