import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ------------------------------
# Data Loading and Initial Inspection
# ------------------------------
filepath = 'toxics_release_inventory.csv'

try:
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Successfully loaded data from '{filepath}'.")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows of the data:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found.")
    print("Please make sure you have uploaded the file to your Colab session.")

# ------------------------------
# Feature Selection and Data Cleaning
# ------------------------------
release_features = [
    'FUGITIVE', 'STACK', 'WATER1', 'LANDFILL', 'POTW',
    'MAX ONSITE', 'OFF STE REL1', 'OFF STE REL2'
]
valid_features = []

for col in release_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        valid_features.append(col)
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping.")

df['TOTAL_RELEASE'] = df[valid_features].sum(axis=1)

print("Successfully cleaned data and created 'TOTAL_RELEASE' feature.")
print(df['TOTAL_RELEASE'].describe())

# ------------------------------
# Creating the Target Variable
# ------------------------------
non_zero_releases = df[df['TOTAL_RELEASE'] > 0]['TOTAL_RELEASE']
low_threshold = non_zero_releases.quantile(0.33)
high_threshold = non_zero_releases.quantile(0.66)

print(f"Defining danger levels with thresholds:")
print(f"  - 'Safe': Total Release <= {low_threshold:.2f}")
print(f"  - 'Warning': Total Release between {low_threshold:.2f} and {high_threshold:.2f}")
print(f"  - 'Danger': Total Release > {high_threshold:.2f}")

def assign_danger_level(total_release):
    if total_release <= low_threshold:
        return 'Safe'
    elif total_release <= high_threshold:
        return 'Warning'
    else:
        return 'Danger'

df['DANGER_LEVEL'] = df['TOTAL_RELEASE'].apply(assign_danger_level)

print(df['DANGER_LEVEL'].value_counts())

# ------------------------------
# Exploratory Data Analysis
# ------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(x='DANGER_LEVEL', data=df, order=['Safe', 'Warning', 'Danger'])
plt.title('Distribution of Danger Levels')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['TOTAL_RELEASE'], bins=50, kde=False)
plt.title('Distribution of Total Chemical Releases')
plt.yscale('log')
plt.show()

# ------------------------------
# Splitting the Data
# ------------------------------
X = df[valid_features]
y = df['DANGER_LEVEL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# ------------------------------
# Base Model Training
# ------------------------------
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)
print("Decision Tree model trained successfully on original data!")

# ------------------------------
# Synthetic Data Generation
# ------------------------------
num_synthetic_rows = 15000
synthetic_data = {}
for feature in valid_features:
    synthetic_data[feature] = X_train[feature].sample(
        n=num_synthetic_rows, replace=True, random_state=42
    ).values

X_synthetic = pd.DataFrame(synthetic_data)
print(f"Generated {num_synthetic_rows} synthetic rows.")

y_synthetic_pred = model.predict(X_synthetic)
print(pd.Series(y_synthetic_pred).value_counts())

synthetic_df = X_synthetic.copy()
synthetic_df['DANGER_LEVEL'] = y_synthetic_pred

# ------------------------------
# Combine Original + Synthetic Data
# ------------------------------
X_train_reset = X_train.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

X_combined = pd.concat([X_train_reset, synthetic_df[valid_features]], ignore_index=True)
y_combined = pd.concat([y_train_reset, synthetic_df['DANGER_LEVEL']], ignore_index=True)

print("Shape of combined features:", X_combined.shape)
print("Shape of combined target:", y_combined.shape)

# ------------------------------
# Train Combined (Synthetic) Model
# ------------------------------
combined_model = DecisionTreeClassifier(random_state=42, max_depth=5)
combined_model.fit(X_combined, y_combined)
print("Decision Tree model trained successfully on combined data!")

# ------------------------------
# Evaluation - Base Model
# ------------------------------
y_pred = model.predict(X_test)
print("\n--- Base Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Base Model)')
plt.show()

# ------------------------------
# Evaluation - Combined Model
# ------------------------------
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

y_combined_pred = combined_model.predict(X_combined_test)
print("\n--- Combined Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_combined_test, y_combined_pred):.4f}")
print(confusion_matrix(y_combined_test, y_combined_pred))
print(classification_report(y_combined_test, y_combined_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_combined_test, y_combined_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Combined Model)')
plt.show()

# ------------------------------
# Comparison Summary
# ------------------------------
print("\n--- Comparison ---")
print(f"Base Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Combined Model Accuracy: {accuracy_score(y_combined_test, y_combined_pred):.4f}")

# ------------------------------
# Export Model
# ------------------------------
model_filename = 'combined_model.joblib'
joblib.dump(combined_model, model_filename)
print(f"Model exported to '{model_filename}'")

# ------------------------------
# Verification: Load & Predict
# ------------------------------
loaded_model = joblib.load(model_filename)
print("Model loaded successfully!")

if 'X_test' in locals():
    sample_data = X_test.iloc[[0]]
    print("Example prediction:", loaded_model.predict(sample_data))

if __name__ == '__main__':
    # Place everything you currently have inside this block.
    # Example:
    print("Training and exporting model...")

