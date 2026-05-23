import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("cleaned_titanic.csv", index=False)

print("\nData Cleaning Completed Successfully!")