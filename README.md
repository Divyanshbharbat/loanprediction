# 🏦 FinQuest AI | Smart Loan Prediction Platform

FinQuest AI is an end-to-end Machine Learning (ML) and Large Language Model (LLM) hybrid application designed to automate and explain credit analysis. It utilizes a Random Forest classifier to evaluate loan eligibility and leverages the Google Gemini API to generate personalized financial advice and decision explanations.

---

## 🚀 Key Features

* **Instant Loan Prediction**: Computes loan approval status and confidence scores based on applicant credentials.
* **Explainable AI**: Integration with Google Gemini (`gemini-2.5-flash`) provides a natural-language explanation of why a loan was approved or rejected.
* **Personalized Recommendations**: Generates tailored financial advice based on applicant solvency parameters.
* **Risk Meter & Debt Analysis**: Displays an interactive Portfolio Risk Index gauge and calculates critical ratios like Debt-to-Income (DTI) and Loan-to-Income.
* **Analytics Registry**: Keeps a log of previous evaluation outcomes and shows insights (outcomes distribution, credit scores distribution, and salary vs. request scatter charts).

---

## 🛠️ Tech Stack

* **UI Framework**: Streamlit (Python Web App Interface)
* **Database**: SQLite3 (Local storage for query auditing)
* **ML Classifier**: Scikit-learn (Random Forest Pipeline, cat-encoders, scalers)
* **AI Engine**: Google Gemini API (`google-genai` SDK)
* **Data Visualization**: Plotly Express & Plotly Graph Objects

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Divyanshbharbat/loanprediction.git
   cd loanprediction
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Google Gemini API Key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the Application**:
   ```bash
   python -m streamlit run app.py
   ```
