import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# File paths
db_path = 'C:/Users/DrVin/Documents/GST/cleaned_data.db'
train_input_path = 'C:/Users/DrVin/Documents/GST/X_Train_Data_Input.csv'
test_input_path = 'C:/Users/DrVin/Documents/GST/X_Test_Data_Input.csv'
train_target_path = 'C:/Users/DrVin/Documents/GST/Y_Train_Data_Target.csv'
test_target_path = 'C:/Users/DrVin/Documents/GST/Y_Test_Data_Target.csv'

# Load and clean data
def load_and_clean_data():
    # Load the datasets
    X_train = pd.read_csv(train_input_path)
    Y_train = pd.read_csv(train_target_path)
    X_test = pd.read_csv(test_input_path)
    Y_test = pd.read_csv(test_target_path)

    # Ensure 'ID' column is not duplicated
    Y_train = Y_train.drop(columns=['ID'])
    Y_test = Y_test.drop(columns=['ID'])

    # Drop rows with missing values in X_train and align Y_train
    combined_train = pd.concat([X_train, Y_train], axis=1)
    combined_train_clean = combined_train.dropna()
    X_train_clean = combined_train_clean.drop(columns=['target'])
    Y_train_clean = combined_train_clean['target']

    # Drop rows with missing values in X_test and align Y_test
    combined_test = pd.concat([X_test, Y_test], axis=1)
    combined_test_clean = combined_test.dropna()
    X_test_clean = combined_test_clean.drop(columns=['target'])
    Y_test_clean = combined_test_clean['target']

    # Save cleaned data to SQLite database
    conn = sqlite3.connect(db_path)
    X_train_clean.to_sql('X_Train_Data', conn, if_exists='replace', index=False)
    Y_train_clean.to_sql('Y_Train_Data', conn, if_exists='replace', index=False)
    X_test_clean.to_sql('X_Test_Data', conn, if_exists='replace', index=False)
    Y_test_clean.to_sql('Y_Test_Data', conn, if_exists='replace', index=False)
    conn.close()

# Feature selection and data preparation
def prepare_data():
    conn = sqlite3.connect(db_path)

    # Load cleaned data from the database
    X_train = pd.read_sql_query("SELECT * FROM X_Train_Data", conn)
    Y_train = pd.read_sql_query("SELECT * FROM Y_Train_Data", conn)['target']
    X_test = pd.read_sql_query("SELECT * FROM X_Test_Data", conn)
    Y_test = pd.read_sql_query("SELECT * FROM Y_Test_Data", conn)['target']

    # Drop non-numeric columns (e.g., 'ID')
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    conn.close()

    return X_train, Y_train, X_test, Y_test

# Model training
def train_model(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    return model

# Model evaluation
def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions)
    
    # Save predictions to a file for comparison
    predictions_df = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions})
    predictions_df.to_csv('C:/Users/DrVin/Documents/GST/predictions.csv', index=False)
    
    return accuracy, report

# Save the trained model
def save_model(model):
    joblib.dump(model, 'C:/Users/DrVin/Documents/GST/predictive_model.pkl')

# Main function to execute all steps
def main():
    # Step 1: Load and clean data
    load_and_clean_data()

    # Step 2: Prepare data for model
    X_train, Y_train, X_test, Y_test = prepare_data()

    # Step 3: Train model
    model = train_model(X_train, Y_train)

    # Step 4: Evaluate model
    accuracy, report = evaluate_model(model, X_test, Y_test)
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Step 5: Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
