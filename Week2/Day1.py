import pandas as pd
import numpy as np

#GroupBy
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

