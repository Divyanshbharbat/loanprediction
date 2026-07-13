import os
from google import genai
from google.genai import errors
from dotenv import load_dotenv
import cv2
# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def get_gemini_client():
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

def generate_explanation(applicant_data, prediction_result, confidence_score):
    client = get_gemini_client()
    if not client:
        return get_fallback_explanation(applicant_data, prediction_result)
        
    prompt = f"""
    You are an expert loan officer and AI explanation system.
    Analyze the following loan application details:
    - Applicant Monthly Income: ${applicant_data['applicant_income']:,}
    - Co-applicant Monthly Income: ${applicant_data['coapplicant_income']:,}
    - Credit Score: {applicant_data['credit_score']} (300-850 scale; scores >= 650 generally represent a good credit history)
    - Dependents: {applicant_data['dependents']}
    - Education: {applicant_data['education']}
    - Self-Employed: {applicant_data['self_employed']}
    - Marital Status: {applicant_data['married']}
    - Requested Loan Amount: ${applicant_data['loan_amount'] * 1000:,} (in absolute dollars)
    - Loan Term: {applicant_data['loan_term']} months
    - Property Area: {applicant_data['property_area']}
    
    The Machine Learning model predicted that the loan is: {prediction_result} (Confidence: {confidence_score:.1f}%).
    
    Please convert this ML result into a simple, professional, and empathetic explanation in human language.
    Explain the primary reasons for this outcome (e.g. referencing credit history/score, income-to-loan ratio, self-employment status, etc.). Keep it concise, friendly, and under 100 words. Do not include boilerplate text or introductory chatter.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except errors.APIError as e:
        print(f"Gemini API Error during explanation generation: {e}")
        return get_fallback_explanation(applicant_data, prediction_result)
    except Exception as e:
        print(f"Unexpected error during explanation generation: {e}")
        return get_fallback_explanation(applicant_data, prediction_result)

def generate_financial_advice(applicant_data, prediction_result):
    client = get_gemini_client()
    if not client:
        return get_fallback_advice(applicant_data, prediction_result)
        
    prompt = f"""
    You are a senior financial advisor.
    Based on the loan applicant's details and prediction:
    - Prediction: {prediction_result}
    - Applicant Monthly Income: ${applicant_data['applicant_income']:,}
    - Co-applicant Monthly Income: ${applicant_data['coapplicant_income']:,}
    - Credit Score: {applicant_data['credit_score']}
    - Requested Loan Amount: ${applicant_data['loan_amount'] * 1000:,}
    - Existing Monthly Debt: ${applicant_data.get('existing_debt', 0):,}
    
    Provide 3 brief, highly actionable financial tips (as bullet points, without headers or numbering) for this applicant.
    If the prediction is REJECTED: advise them on how to improve their financial health to successfully reapply in the future (e.g. debt reduction, boosting credit score, raising down payment).
    If the prediction is APPROVED: advise them on how to manage their loan responsibly (e.g. setting up automatic payments, budgeting, maintaining good credit).
    Make the tips concise and direct. Do not write any intro or outro.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except errors.APIError as e:
        print(f"Gemini API Error during advice generation: {e}")
        return get_fallback_advice(applicant_data, prediction_result)
    except Exception as e:
        print(f"Unexpected error during advice generation: {e}")
        return get_fallback_advice(applicant_data, prediction_result)

def get_fallback_explanation(applicant_data, prediction_result):
    score = applicant_data['credit_score']
    income = applicant_data['applicant_income'] + applicant_data['coapplicant_income']
    loan = applicant_data['loan_amount'] * 1000
    
    if prediction_result == "Approved":
        return (f"Based on your strong credit profile (Score: {score}) and a solid combined monthly income of "
                f"${income:,}, the loan request of ${loan:,} falls within a safe debt-to-income threshold, "
                f"resulting in a high approval probability.")
    else:
        reasons = []
        if score < 650:
            reasons.append(f"a lower credit rating (Score: {score})")
        if income * 10 < loan:
            reasons.append("a high loan-to-income ratio")
        
        reasons_str = " and ".join(reasons) if reasons else "general risk parameters"
        return (f"Your loan request of ${loan:,} was rejected. This is primarily due to {reasons_str}. "
                f"Lenders require a stronger credit rating or higher income relative to the requested loan amount "
                f"to mitigate credit default risks.")

def get_fallback_advice(applicant_data, prediction_result):
    if prediction_result == "Approved":
        return """* **Set up Auto-Pay:** Automate your monthly loan repayments to ensure you never miss a deadline and avoid late fees.
* **Maintain Your Credit Score:** Keep your other credit cards and debts in good standing; do not take on new loans during this loan term.
* **Emergency Buffer:** Keep at least 3-6 months of loan installments saved in an emergency account to cover unexpected financial disruptions."""
    else:
        return """* **Boost Credit Score:** Review your credit file and pay down active revolving balances to improve your score to at least 650.
* **Reduce Debt-to-Income (DTI):** Settle existing high-interest debts or co-sign with a co-applicant to increase combined income safety.
* **Lower Loan Request:** Consider requesting a lower principal amount (e.g. 20% less) or extending the term to lower monthly installments."""
