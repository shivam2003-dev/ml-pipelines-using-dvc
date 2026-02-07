import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os
import yaml

with open("params.yml", "r") as f:
    params = yaml.safe_load(f) or {}

model_params = (params.get("model_training") or {})

# fetch the data
train_data = pd.read_csv("data/features/train_bow.csv")

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

clf = GradientBoostingClassifier(
    n_estimators=int(model_params.get("n_estimators", 50)),
    learning_rate=float(model_params.get("learning_rate", 0.1)),
    random_state=int(model_params.get("random_state", 42)),
)

clf.fit(X_train, y_train)

models_dir = os.path.join("models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(clf, os.path.join(models_dir, "model.joblib"))
