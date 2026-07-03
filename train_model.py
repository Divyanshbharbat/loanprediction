import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def clean_and_train():
    # 1. Load the dataset
    print("Loading dataset train.csv...")
    df = pd.read_csv("train.csv")
    
    # 2. Preprocess target and features
    # Drop Loan_ID as it is just an identifier
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
        
    # Clean target Loan_Status: Y -> 1, N -> 0
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    
    # Clean Dependents column: replace '3+' with '3' and convert to float (but treat as category to handle it safely)
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    
    # Fill target NaNs if any (should not be there, but let's drop them just in case)
    df = df.dropna(subset=["Loan_Status"])
    
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    
    # Define categorical and numerical features
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
    numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    
    # Ensure categorical columns are strings to avoid dtype issues during imputation/encoding
    for col in categorical_cols:
        X[col] = X[col].astype(str).replace("nan", np.nan)
        
    # 3. Create Preprocessing Transformers
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    # 4. Define full pipeline (Preprocessor + Model)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8))
    ])
    
    # 5. Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Train the model pipeline
    print("Training Random Forest model pipeline...")
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluate performance
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    
    # 8. Save the model pipeline
    model_filename = "loan_pipeline.pkl"
    print(f"Saving trained pipeline to {model_filename}...")
    with open(model_filename, "wb") as f:
        pickle.dump(pipeline, f)
    
    print("Model pipeline training and serialization completed successfully!")

if __name__ == "__main__":
    clean_and_train()
