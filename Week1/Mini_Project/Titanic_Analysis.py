import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = sns.load_dataset("titanic")

# First look at the data
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

#Explore & Understand the Data
# Basic information
print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

#Clean the Data
# Fill missing age with median age
df["age"] = df["age"].fillna(df["age"].median())

# Fill missing embarked with most common value
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Drop columns we don't need
df = df.drop(["deck", "embark_town", "alive", "who", "adult_male"], axis=1)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nCleaned shape:", df.shape)

#Analysis & Visualization
# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Plot 1 — Overall Survival Count
plt.subplot(2, 3, 1)
df["survived"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Survival Count")
plt.xlabel("0 = Died, 1 = Survived")
plt.ylabel("Count")
plt.xticks(rotation=0)

# Plot 2 — Survival by Gender
plt.subplot(2, 3, 2)
sns.barplot(x="sex", y="survived", data=df)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

# Plot 3 — Survival by Passenger Class
plt.subplot(2, 3, 3)
sns.barplot(x="pclass", y="survived", data=df)
plt.title("Survival Rate by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

# Plot 4 — Age Distribution
plt.subplot(2, 3, 4)
df["age"].hist(bins=30, color="steelblue", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

# Plot 5 — Fare Distribution
plt.subplot(2, 3, 5)
df["fare"].hist(bins=30, color="orange", edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")

# Plot 6 — Correlation Heatmap
plt.subplot(2, 3, 6)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("titanic_analysis.png")  # saves the plot as image
plt.show()

#Key Insights
print("=" * 50)
print("KEY INSIGHTS FROM TITANIC DATASET")
print("=" * 50)

# Overall survival rate
survival_rate = df["survived"].mean() * 100
print(f"\n1. Overall Survival Rate: {survival_rate:.1f}%")

# Survival by gender
print("\n2. Survival Rate by Gender:")
print(df.groupby("sex")["survived"].mean() * 100)

# Survival by class
print("\n3. Survival Rate by Passenger Class:")
print(df.groupby("pclass")["survived"].mean() * 100)

# Average age
print(f"\n4. Average Age of Passengers: {df['age'].mean():.1f} years")

# Average fare
print(f"\n5. Average Fare Paid: ${df['fare'].mean():.2f}")

print("\n" + "=" * 50)
print("CONCLUSIONS:")
print("=" * 50)
print("- Women had significantly higher survival rates than men")
print("- 1st class passengers survived more than 2nd and 3rd class")
print("- Younger passengers had slightly better survival rates")
print("- Higher fare correlated with better survival chances")
