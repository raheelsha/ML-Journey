import pandas as pd
import numpy as np

#Concept 1 GroupBy
data = {
    "name": ["Raheel", "Ali", "Sara", "Ahmed", "Usman", "Ayesha"],
    "department": ["AI", "AI", "Data", "Data", "AI", "Data"],
    "salary": [50000, 60000, 55000, 65000, 70000, 58000],
    "experience": [1, 2, 1, 3, 2, 2]
}
df=pd.DataFrame(data)
#Average Salary by Department
print(df.groupby("department")["salary"].mean())
# Multiple statistics at once
print(df.groupby("department")["salary"].agg(["mean", "min", "max"]))
#Group by departments and get all details
print(df.groupby("department").describe())

#Concept 2 Sorting & Ranking
print(df.sort_values("salary", ascending=False))
# Sort by multiple columns
print(df.sort_values(["department", "salary"], ascending=[True, False]))
# Rank employees by salary
df["salary_rank"] = df["salary"].rank(ascending=False)
print(df)

#Concept 3 Merging DataFrames
# Employee basic info
employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name": ["Raheel", "Ali", "Sara", "Ahmed"],
    "dept_id": [101, 102, 101, 103]
})

# Department info
departments = pd.DataFrame({
    "dept_id": [101, 102, 103],
    "dept_name": ["AI", "Data Science", "MLOps"]
})

# Merge them together
merged = pd.merge(employees, departments, on="dept_id")
print(merged)

#Concept 4 - Apply Function
data = {
    "name": ["Raheel", "Ali", "Sara", "Ahmed"],
    "salary": [50000, 60000, 55000, 65000]
}
df = pd.DataFrame(data)
def characterized_salary(salary):
    if salary>=60000:
        return "High"
    elif salary>=55000:
        return "Medium"
    else:
        return "Low" 
    
df["Salary_Category"]=df["salary"].apply(characterized_salary)
print(df)

#Lambda Version
df["salary_usd"]=df["salary"].apply(lambda x:x/280)
print(df)
df["salary_aus$"]=df["salary"].apply(lambda x:x/190)
print(df)

#Concept 5 - Pivot Table
data = {
    "name": ["Raheel", "Ali", "Sara", "Ahmed", "Usman", "Ayesha"],
    "department": ["AI", "AI", "Data", "Data", "AI", "Data"],
    "month": ["Jan", "Feb", "Jan", "Feb", "Jan", "Feb"],
    "sales": [5000, 6000, 4500, 7000, 5500, 6500]
}

df = pd.DataFrame(data)
pivot = df.pivot_table(
    values="sales",
    index="department",
    columns="month",
    aggfunc="mean"
)
print(pivot)

#Concept 6 — Working with CSV
df.to_csv("employee.csv", index=False)
# Read it back
df_loaded = pd.read_csv("employee.csv")
print(df_loaded.head())

#Excercises
data = {
    "name": ["Raheel", "Ali", "Sara", "Ahmed", "Usman", "Ayesha","Aqeel","Malik"],
    "university": ["Pu","UCP","Lums","NUML","VU","VUP","AIR","ETC"],
    "marks": [100,80,90,50,30,20,10,90],
    "attendence": [9, 10, 9, 2, 1, 2,3,4]
}
df=pd.DataFrame(data)
average_stats = df.groupby('university')[['marks', 'attendence']].mean()

print(average_stats)
# 2
students = pd.DataFrame({
    "name": ["Raheel", "Ali", "Sara", "Ahmed"],
    "course_id": [101, 102, 101, 103]
})

# Department info
departments = pd.DataFrame({
    "course_id": [101, 102, 103],
    "course_name": ["AI", "Data Science", "MLOps"]
})

# Merge them together
merged = pd.merge(students, departments, on="course_id")
print(merged)

#3
import pandas as pd

# Create DataFrame
data = {
    'Employee': ['Ali', 'Sara', 'Zain', 'Hina', 'Bilal'],
    'Salary_PKR': [120000, 75000, 95000, 60000, 150000]
}
df = pd.DataFrame(data)

# 1. Convert Salary to USD (divide by 280)
df['Salary_USD'] = df['Salary_PKR'].apply(lambda x: x / 280)

# 2. Add Category column
df['Category'] = df['Salary_PKR'].apply(lambda x: 'Senior' if x > 80000 else 'Junior')

print(df)

#4
import seaborn as sns

# Load dataset
titanic = sns.load_dataset("titanic")

# 1. Survival rate by gender
survival_gender = titanic.groupby('sex')['survived'].mean()

# 2. Survival rate by passenger class
survival_class = titanic.groupby('pclass')['survived'].mean()

print("Survival Rate by Gender:\n", survival_gender)
print("\nSurvival Rate by Class:\n", survival_class)

