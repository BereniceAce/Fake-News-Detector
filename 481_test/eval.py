import pandas as pd
from model import predict
from features import extract_features

# Load datasets safely
try:
    true_df = pd.read_csv("true.csv")
    fake_df = pd.read_csv("fake.csv")
except FileNotFoundError:
    print("Error: true.csv or fake.csv not found in the current folder.")
    exit()
except PermissionError:
    print("Error: Permission denied. Close any programs using the CSV files and try again.")
    exit()

# Ensure there is a 'text' column in your CSVs
if 'text' not in true_df.columns or 'text' not in fake_df.columns:
    print("Error: CSV files must contain a 'text' column.")
    exit()

# Combine datasets and create labels
true_df['label'] = 'Real'
fake_df['label'] = 'Fake'
all_df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the dataset
all_df = all_df.sample(frac=1).reset_index(drop=True)

# Counters for evaluation
TP = FP = TN = FN = 0

for _, row in all_df.iterrows():
    text = row['text']
    actual = row['label']
    result = predict(text)
    predicted = result['verdict']

    if actual == "Fake" and predicted == "Fake":
        TP += 1
    elif actual == "Fake" and predicted == "Real":
        FN += 1
    elif actual == "Real" and predicted == "Real":
        TN += 1
    elif actual == "Real" and predicted == "Fake":
        FP += 1

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Evaluation Metrics:")
print(f"Total articles: {len(all_df)}")
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
