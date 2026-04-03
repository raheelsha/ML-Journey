import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Concept 1 — Central Tendency
scores = [45, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200]
# Notice 200 is an outlier!
mean=np.mean(scores)
print("Mean:",{mean})
median=np.median(scores)
print("Median:",{median})

print(f"\nMean is pulled up by outlier: {mean}")
print(f"Median stays stable: {median}")
print("For skewed data always use median!")

#Concept 2 Spread of the Data

salaries = [30000, 35000, 40000, 45000, 50000, 55000, 60000, 200000]
variance=np.var(salaries)
std_dev=np.std(salaries)
print(f"Variance:", {variance})
print("Standard Daviation:", std_dev)
print("Minimum Salary:",np.min(salaries))
print("Maximum Salary:",np.max(salaries))
print("Range:", np.max(salaries) - np.min(salaries))
print(f"n/25th Percentile:", {np.percentile(salaries,25)})
print(f"n/25th Percentile:", {np.percentile(salaries,50)})
print(f"n/25th Percentile:", {np.percentile(salaries,75)})
print(f"n/25th Percentile:", {np.percentile(salaries,100)})

#Concept 3 — Normal Distribution
np.random.seed(40)  
data= np.random.normal(
    loc=70,
    scale=10,
    size=1000
)
plt.figure(figsize=(10, 4)) #width=10incehs, height=4 inches
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, color="steelblue", edgecolor="black")
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.axvline(np.mean(data), color="red", linestyle="--", label="Mean") #axvline(...): Draws a vertical red dashed line exactly at the mathematical average.
plt.legend()
plt.subplot(1, 2, 2)
sns.kdeplot(data, color="steelblue", fill=True)
plt.title("Density Plot")
plt.xlabel("Value")
plt.tight_layout()
plt.show()

print(f"Mean: {np.mean(data):.2f}")
print(f"Std Dev: {np.std(data):.2f}")

#concept 4 — Correlation
np.random.seed(48)
data = {
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "marks":       [45, 50, 55, 65, 70, 75, 80, 85, 90, 95],
    "sleep_hours": [8, 7, 7, 6, 6, 7, 5, 5, 4, 4],
    "phone_hours": [6, 5, 5, 4, 3, 3, 2, 2, 1, 1]
}
df = pd.DataFrame(data)
corr =df.corr()
print("Correlation Matrix:")
print(corr)
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Concept 5 — Outlier Detection
salaries = [30000, 32000, 35000, 33000, 31000, 
            34000, 36000, 32000, 500000, 29000]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.boxplot(salaries)
plt.title("BoxPlot-Outlier Detection")
plt.ylabel("Salary")
plt.show()

#IQR(Inter Quartile Range)
plt.subplot(1,2,1)
data=np.array(salaries)
Q1=np.percentile(data,25)
Q3=np.percentile(data,75)
IQR=Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

outliers=data[(data<lower_bound) | (data>upper_bound)]
print(f"Outliers Detected :{outliers}")

plt.hist(salaries, bins=10, color="steelblue", edgecolor="black")
plt.title("Histogram with Outliers")
plt.xlabel("Salary")
plt.show()

#Concept 6 — Real World Statistics on Titanic
df=sns.load_dataset("titanic")
print("=== TITANIC STATISTICAL ANALYSIS ===\n")
print(f"Mean Age: {df['age'].mean():.1f}")
print(f"Median Age: {df['age'].median():.1f}")
print(f"Standard Dev Age: {df['age'].std():.1f}")

# Fare analysis
print("\nFare Statistics:")
print(f"Mean Fare: ${df['fare'].mean():.2f}")
print(f"Median Fare: ${df['fare'].median():.2f}")
print(f"Max Fare: ${df['fare'].max():.2f}")

# Correlation with survival
print("\nCorrelation with Survival:")
numeric_df = df[["survived", "pclass", "age", "fare"]].dropna()
print(numeric_df.corr()["survived"].sort_values(ascending=False))

#Excercises # 1

salaries = [45000, 48000, 50000, 52000, 55000, 58000, 60000, 
            62000, 65000, 70000, 72000, 75000, 80000, 1500000, 2500000]

mean=np.mean(salaries)
print("Mean:",{mean})
median=np.median(salaries)
print("Median:",{median})
std_dev=np.std(salaries)
print("Standard Deviation:", std_dev)
print("In this scenario, the Median (62,000.0) is absolutely the most reliable metric.")

# Exercise # 2
data = {
    "hours_studied": [8, 2, 5, 7, 4, 1, 10, 3],
    "marks":         [95, 45, 75, 85, 65, 50, 98, 55],
    "attendance":    [95, 65, 80, 85, 90, 95, 100, 75],
    "mobile_usage":  [2, 7, 4, 3, 6, 8, 1, 5]
}
df = pd.DataFrame(data)
corr =df.corr()
print("Correlation Matrix:")
print(corr)
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
print("Hours Studied has the strongest positive correlation with marks (0.95), while Mobile Usage has a strong negative correlation (-0.85).")


# Exercise 3: Outlier Detection (IQR Method)
# ==========================================
# 1. Create a dataset of exam scores with 3 obvious outliers
np.random.seed(42)
scores = np.random.normal(loc=75, scale=8, size=50).tolist() # Normal scores
scores.extend([12, 18, 140]) # 2 extremely low, 1 extremely high outlier
df_scores = pd.DataFrame({'scores': scores})

# 2. Use the IQR method to detect outliers
Q1 = df_scores['scores'].quantile(0.25)
Q3 = df_scores['scores'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# Filter to find outliers
outliers = df_scores[(df_scores['scores'] < lower_fence) | (df_scores['scores'] > upper_fence)]

print("--- Exercise 3: Outlier Detection ---")
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Lower Fence: {lower_fence:.2f}, Upper Fence: {upper_fence:.2f}")
print(f"Detected Outliers:\n{outliers}\n")

# 3. Show a boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_scores['scores'], color="lightcoral")
plt.title("Exercise 3: Boxplot of Exam Scores with Outliers")
plt.xlabel("Exam Scores")
plt.savefig("exercise3_boxplot.png")
plt.close()


# ==========================================
# Exercise 4: Normal Distribution
# ==========================================
# 1. Generate 500 random heights (mean=170, std=8)
heights = np.random.normal(loc=170, scale=8, size=500)

# 2. Print mean and standard deviation
print("--- Exercise 4: Normal Distribution ---")
print(f"Calculated Mean: {np.mean(heights):.2f}")
print(f"Calculated Std Dev: {np.std(heights):.2f}")

# 3. Plot a histogram with the mean line
plt.figure(figsize=(8, 5))
sns.histplot(heights, bins=30, color="steelblue", kde=False, edgecolor="black")
plt.axvline(np.mean(heights), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(heights):.2f}")
plt.title("Exercise 4: Histogram of Heights")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("exercise4_histogram.png")
plt.close()