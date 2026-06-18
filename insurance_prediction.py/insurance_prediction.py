import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("insurance.csv")

# Convert text columns to numbers
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df["region"] = le.fit_transform(df["region"])

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# New customer prediction
new_data = [[25, 1, 28.5, 1, 0, 2]]
prediction = model.predict(new_data)

print("Predicted Insurance Claim Amount:", prediction[0])

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["charges"], kde=True)
plt.title("Insurance Charges Distribution")
plt.show()

numeric_df = df.select_dtypes(include='number')

sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()