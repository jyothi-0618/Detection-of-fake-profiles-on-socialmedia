import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load training data
train_path = 'train.csv'
test_path = 'test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# Standardize features
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Perform PCA (optional, remove if not needed)
pca = PCA(n_components=2)
x_train_sc_pca = pca.fit_transform(x_train_sc)
x_test_sc_pca = pca.transform(x_test_sc)

# Train Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rfc.fit(x_train_sc, y_train)

# Save the trained model and the scaler
joblib.dump(rfc, "rfc_model.pkl")
joblib.dump(sc, "scaler.pkl")

print("Model and scaler saved successfully.")
