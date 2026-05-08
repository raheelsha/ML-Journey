import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create Dataset
np.random.seed(42)
experience =np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
salary = experience * 8000 + np.random.randint(-5000, 5000, 20) + 25000
df = pd.DataFrame({"experience":experience, "salary":salary})
print("DataSet:")
print(df.head())

# Step 1 — Define features and target
X = df[["experience"]]
y = df[["salary"]]

# Step 2 — Split data

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"\nTesting samples:{len(X_test)}")

# Step 3 — Create and train model
model= LinearRegression()
model.fit(X_train , y_train)

# Step 4 — Make predictions
predictions=model.predict(X_test)

# Step 5 — Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print("R² close to 1.0 means excellent model!")

# Step 6 — Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, predictions, color="red", label="Predicted")
plt.title("Linear Regression — Experience vs Salary")
plt.xlabel("Experience (years)")
plt.ylabel("Salary (PKR)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, predictions, color="green")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--")
plt.title("Actual vs Predicted")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.tight_layout()
plt.show()

# Predict new salary
new_experience = [[15]]
predicted_salary = model.predict(new_experience)
# Adding [0][0] or using .item() gets the raw number out of the array
print(f"\nPredicted salary for 12 years experience: PKR {predicted_salary[0][0]:,.0f}")

#=====================================================================================================================#

# Concept 2 — Logistic Regression (Classification)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Create dataset — predict pass/fail based on study hours and attendance
np.random.seed(42)
n=100
study_hours= np.random.randint(1,10,n)
attendance=np.random.randint(50,100,n)
# Pass if study hours > 5 and attendance > 75
passed=((study_hours>5) &(attendance>75).astype(int))

df = pd.DataFrame({
    "attendance": attendance,
    "study_hours":study_hours,
    "passed":passed
})
print("Dataset Sample:")
print(df.head(10))
print(f"\n passed rate:{passed.mean()*100:.1f}%")

# Features and target
X = df[["study_hours", "attendance"]]
y = df["passed"]

# Split
X_train,X_test, y_train, y_test=train_test_split(
    X,y, test_size=0.2 ,random_state=42
)
# Train
model=LogisticRegression()
model.fit(X_train, y_train)
#Prediction
predictions= model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy*100:.1f}%")

#Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("\nConfusion Matrix explained:")
print(f"True Negatives (correctly predicted Fail): {cm[0][0]}")
print(f"False Positives (predicted Pass but actually Fail): {cm[0][1]}")
print(f"False Negatives (predicted Fail but actually Pass): {cm[1][0]}")
print(f"True Positives (correctly predicted Pass): {cm[1][1]}")

#=====================================================================================================================#

# Concept 3 — Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age", "fare"]].dropna()

# Convert sex to numbers
le =LabelEncoder()
df["sex"]=le.fit_transform(df["sex"])
# female=0, male=1

print("Prepared Dataset:")
print(df.head())

# Features and target
X=df[["pclass","sex","age","fare"]]
y=df[["survived"]]

# Split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2 ,random_state=42
)

# Train Decision Tree
model =DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict
predict=model.predict(X_test)
accuracy= accuracy_score(y_test, predictions)
print(f"\nDecision Tree Accuracy: {accuracy*100:.1f}%")

# Feature Importance — very useful in ML!
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(8, 4))
sns.barplot(x="importance", y="feature", data=feature_importance)
plt.title("Feature Importance — Titanic Survival")
plt.show()
