import numpy as np
import pandas as pd

import json
import os
import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report

clf = joblib.load("models/model.joblib")
test_data = pd.read_csv("data/features/test_bow.csv")

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'f1': f1,
    'auc':auc
}

reports_dir = os.path.join("reports")
os.makedirs(reports_dir, exist_ok=True)

with open(os.path.join(reports_dir, "metrics.json"), "w") as file:
    json.dump(metrics_dict, file, indent=4)

with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred))
