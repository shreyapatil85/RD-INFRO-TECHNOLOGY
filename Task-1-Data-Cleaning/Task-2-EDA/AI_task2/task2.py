import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_titanic.csv")

print(df.head())

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.countplot(x='Sex', data=df)
plt.title("Male vs Female")
plt.show()

sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.show()

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

print("EDA Completed Successfully!")