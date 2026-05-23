import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("cleaned_titanic.csv")

print("Original Data:")
print(df.head())

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

df["Embarked"] = le.fit_transform(df["Embarked"])

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\nFeature Engineered Data:")
print(df.head())

df.to_csv("feature_engineered_titanic.csv", index=False)

print("\nFeature Engineering Completed Successfully!")