import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

DB_FILE = "loan_system.db"

def get_connection():
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create predictions history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            gender TEXT,
            married TEXT,
            dependents TEXT,
            education TEXT,
            self_employed TEXT,
            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_term REAL,
            credit_score INTEGER,
            property_area TEXT,
            prediction_result TEXT,
            confidence_score REAL,
            llm_explanation TEXT,
            ai_advice TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    
    conn.commit()
    conn.close()

def _hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username, password, full_name):
    username = username.strip().lower()
    if not username or not password or not full_name:
        return False, "All fields are required."
        
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if user already exists
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "Username already exists."
            
        # Hash password and insert
        pw_hash = _hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            (username, pw_hash, full_name)
        )
        conn.commit()
        return True, "Registration successful!"
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def verify_user(username, password):
    username = username.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        pw_hash = _hash_password(password)
        cursor.execute(
            "SELECT username, full_name FROM users WHERE username = ? AND password_hash = ?",
            (username, pw_hash)
        )
        row = cursor.fetchone()
        if row:
            return {"username": row[0], "full_name": row[1]}
        return None
    except Exception as e:
        print(f"Error during verification: {e}")
        return None
    finally:
        conn.close()

def save_prediction(username, gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount, loan_term,
                    credit_score, property_area, prediction_result, confidence_score,
                    llm_explanation, ai_advice):
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO predictions (
                username, gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount, loan_term,
                credit_score, property_area, prediction_result, confidence_score,
                llm_explanation, ai_advice
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username.lower() if username else None, gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_score, property_area, prediction_result, confidence_score,
            llm_explanation, ai_advice
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False
    finally:
        conn.close()

def get_user_history(username):
    conn = get_connection()
    try:
        query = "SELECT * FROM predictions WHERE username = ? ORDER BY created_at DESC"
        df = pd.read_sql_query(query, conn, params=(username.lower(),))
        return df
    except Exception as e:
        print(f"Error fetching history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Initialize tables immediately on import
init_db()
