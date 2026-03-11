#Creating a DataFrame
import pandas as pd
Data={
    "name":["Raheel","Shaukat","Sara","Malik"],
    "Age":[25,50,20,20],
    "city":["Lahore","Karachi","Islamabad","Peshawar"],
    "score":[50,80,90,100]
}
df=pd.DataFrame(Data)
df=pd.DataFrame(Data,index=["Day1","Day2","Day3","Day4"])   #Used to give the name of the days to the data
print(df)

#Exploring Your Data
print(df.head())        # first 5 rows
print(df.tail())        # last 5 rows
print(df.shape)         # (rows, columns)
print(df.columns)       # column names
print(df.info())        # data types and null values
print(df.describe())    # statistics summary

#Selecting Data
print(df["name"])
print(df[["name","Age"]])
print(df.iloc[0])       #used to select specific row/column
print(df.iloc[1:3])

#Adding & Modifying Columns
df["passed"]=df["score"]>50
print(df)
df["score"]=df["score"]+5
print(df)

df["grade"] = ["A" if s >= 90 else "B" if s >= 80 else "C" for s in df["score"]]    # Add a grade column based on score
print(df)

# Interesting Excercises For Practise
students={
    'names':['raheel','adeel','aqeel',"abrar"],
    'Score':[80,70,60,50],
    'subjects':['english','urdu','math','computer'],
    'attendence':[80,90,70,75]
}
df=pd.DataFrame(students)
print(df)
print(df.shape)
print(df.info())
print(df.describe())
#Filter & Select
df["passed"]=df["Score"]>75
print(df)
df["Totalattendence"]=df["attendence"]>80
print(df)
print(df.iloc[:,0])
print(df.iloc[:,1])

df["result"] = ["Pass" if s >= 50 else "Fail" for s in df["Score"]]    # Add a result column based on score
print(df)

data = {
    'Student': ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 5'],
    'Marks': [85, 90, None, 70, None]
}
df = pd.DataFrame(data)

print("--- Before Filling ---")
print(df)

# 2. Calculate the average of the existing marks
# Note: mean() automatically ignores None/NaN values
average_marks = df['Marks'].mean()

# 3. Fill the missing values
df['Marks'] = df['Marks'].fillna(average_marks)

print("\n--- After Filling (with average) ---")
print(df)

