import numpy as np
import pandas as pd
import matplotlib
import seaborn
import sklearn
import sqlite3
import os

# Connect to database (creates it if doesn't exist)
conn = sqlite3.connect("ml_students.db")
cursor=conn.cursor()
# Create a table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        university TEXT,
        marks REAL,
        attendance INTEGER,
        city TEXT
    )
""")

conn.commit()
print("Database and table created successfully!")

# Insert multiple students
students_data = [
    (1, "Raheel", "VUP", 85.5, 90, "Lahore"),
    (2, "Ali", "FAST", 92.0, 85, "Karachi"),
    (3, "Sara", "LUMS", 78.5, 95, "Lahore"),
    (4, "Ahmed", "NUST", 88.0, 80, "Islamabad"),
    (5, "Usman", "VUP", 72.0, 75, "Lahore"),
    (6, "Ayesha", "FAST", 95.0, 92, "Karachi"),
    (7, "Bilal", "LUMS", 65.0, 70, "Islamabad"),
    (8, "Hina", "NUST", 90.0, 88, "Islamabad")
]

cursor.executemany("""
    INSERT OR IGNORE INTO students 
    VALUES (?, ?, ?, ?, ?, ?)
""", students_data)

conn.commit()
print("Data inserted successfully!")

# Concept-3 Select all data
result= cursor.execute("SELECT * FROM students")
rows= result.fetchall()
print("All Students:")
for row in rows:
    print(row)

# Select specific columns
result=cursor.execute("SELECT name, marks FROM students")
print("\n Names and Marks only:")
for rows in result.fetchall():
    print(row)

# Load directly into Pandas DataFrame (most common in ML!)
df = pd.read_sql("SELECT * FROM students",conn)
print("\n As Data Frames:")
print(df)

#Concept 4 — WHERE (Filtering)
df=pd.read_sql(""" SELECT * FROM students WHERE marks >85""",conn)
print(" Students with marks greater than 85:")
print(df)

# Students from Lahore
df=pd.read_sql("""SELECT * FROM students WHERE city='Lahore'""",conn)
print("Student who resides in the Lahore: ")
print(df)

#OR statement
df=pd.read_sql("""SELECT * FROM students WHERE city='Lahore' OR city = 'Karachi' """, conn)
print("Students from the Lahore or Karachi:")
print(df)

#SQL Vs Pandas Comparison
# These do the same thing!

# SQL way
df = pd.read_sql("SELECT * FROM students WHERE marks > 85", conn)

#Pandas Way
df[df["marks"]>85]

#Concept 5 — ORDER BY & LIMIT
# Sort by marks highest to lowest
df =pd.read_sql("""SELECT * FROM students ORDER BY marks ASC """,conn)
print("Students ordered by their marks:")
print(df)
#Comments: For Descending us the "DESC" and for Ascending "ASC"
df =pd.read_sql("""SELECT * FROM students ORDER BY marks DESC LIMIT 3 """,conn)
print("Students ordered by their marks:")
print(df)

# Bottom 3 students
df = pd.read_sql("""
    SELECT * FROM students 
    ORDER BY marks ASC
    LIMIT 3
""", conn)
print("\nBottom 3 students:")
print(df)

#Concept 6 — GROUP BY & Aggregate Functions
df = pd.read_sql("""
    SELECT university, 
           AVG(marks) as avg_marks,
           COUNT(*) as total_students,
           MAX(marks) as highest_marks,
           MIN(marks) as lowest_marks
    FROM students
    GROUP BY university
    ORDER BY avg_marks DESC
""", conn)
print("Stats per University:")
print(df)

# Average marks per city
df=pd.read_sql("""
               SELECT city,
               AVG(marks) as avg_marks,
               COUNT(*) as total_students FROM students
    GROUP BY city
""", conn)
print("\nStats per City:")
print(df)

#Concept 7 — JOIN (Combining Tables)
# Create a second table for courses
# Create a second table for courses
cursor.execute("""
    CREATE TABLE IF NOT EXISTS courses (
        student_id INTEGER,
        course_name TEXT,
        grade TEXT
    )
""")

courses_data = [
    (1, "Machine Learning", "A"),
    (2, "Deep Learning", "A+"),
    (3, "NLP", "B+"),
    (4, "Computer Vision", "A"),
    (5, "Machine Learning", "B"),
    (6, "Deep Learning", "A+"),
]

cursor.executemany("""
    INSERT OR IGNORE INTO courses VALUES (?, ?, ?)
""", courses_data)
conn.commit()

# JOIN both tables
df = pd.read_sql("""
    SELECT students.name, students.marks, 
                  courses.course_name, courses.grade
    FROM students
    JOIN courses ON students.id = courses.student_id
""", conn)
print("Students with their courses:")
print(df)

#Concept 8 — SQL to Pandas (Real ML Workflow)
print("=== Real ML Workflow ===")
df=pd.read_sql("SELECT * FROM students",conn)
print("Step_1 Data Extracted From the Database")
print(df.head())
# Step 2 — Basic analysis
print("\nStep 2 — Quick analysis:")
print(f"Total Students:{len(df)}")
print(f"Average Marks:{df["marks"].mean():.1f}")
print(f"Top performer: {df.loc[df['marks'].idxmax(), 'name']}")
# Step 3 - Ready for ML
X=df[["attendance"]]
Y=df["marks"]
print("\n Data Ready For ML Model!")
print("\n Data Ready For ML Model!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {Y.shape}")
conn.close()

#Excercises
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# EXERCISE 1: Create & Insert
# ==========================================
# 1. Create the connection (this creates company.db in your folder)
conn = sqlite3.connect('company.db')
cursor = conn.cursor()

# 2. Create the employees table
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL,
    experience INTEGER,
    city TEXT
)
''')

# 3. Insert at least 10 employees (Using executemany for efficiency)
# I included Lahore specifically since we need it for Exercise 2!
employee_data = [
    (1, 'Ali', 'IT', 75000, 4, 'Lahore'),
    (2, 'Zainab', 'HR', 45000, 2, 'Karachi'),
    (3, 'Bilal', 'IT', 85000, 5, 'Islamabad'),
    (4, 'Fatima', 'Finance', 95000, 7, 'Lahore'),
    (5, 'Omar', 'Marketing', 55000, 3, 'Lahore'),
    (6, 'Ayesha', 'IT', 48000, 1, 'Peshawar'),
    (7, 'Hassan', 'Finance', 62000, 4, 'Karachi'),
    (8, 'Zara', 'HR', 70000, 6, 'Islamabad'),
    (9, 'Usman', 'Marketing', 40000, 1, 'Lahore'),
    (10, 'Sara', 'IT', 110000, 8, 'Karachi')
]

# Clear table first just in case you run this cell multiple times
cursor.execute('DELETE FROM employees') 
cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)', employee_data)
conn.commit()
print("Exercise 1 Complete: Database and employees created.\n")


# ==========================================
# EXERCISE 2: SELECT & WHERE
# ==========================================
print("--- EXERCISE 2: WHERE Clauses ---")

# Salary > 60000
query_high_salary = "SELECT * FROM employees WHERE salary > 60000"
print("Salary > 60k:\n", pd.read_sql(query_high_salary, conn), "\n")

# From Lahore only
query_lahore = "SELECT * FROM employees WHERE city = 'Lahore'"
print("Lahore Employees:\n", pd.read_sql(query_lahore, conn), "\n")

# Experience > 2 AND Salary > 50000
query_exp_sal = "SELECT * FROM employees WHERE experience > 2 AND salary > 50000"
print("Exp > 2 AND Sal > 50k:\n", pd.read_sql(query_exp_sal, conn), "\n")


# ==========================================
# EXERCISE 3: GROUP BY
# ==========================================
print("--- EXERCISE 3: Aggregations ---")

query_avg_sal = "SELECT department, AVG(salary) as average_salary FROM employees GROUP BY department"
print("Average Salary per Dept:\n", pd.read_sql(query_avg_sal, conn), "\n")

query_total_city = "SELECT city, COUNT(*) as total_employees FROM employees GROUP BY city"
print("Total Employees per City:\n", pd.read_sql(query_total_city, conn), "\n")

query_max_sal = "SELECT department, MAX(salary) as highest_salary FROM employees GROUP BY department"
print("Highest Salary per Dept:\n", pd.read_sql(query_max_sal, conn), "\n")


# ==========================================
# EXERCISE 4: JOIN
# ==========================================
print("--- EXERCISE 4: JOINS ---")

# Create projects table
cursor.execute('''
CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    project_name TEXT,
    status TEXT
)
''')

project_data = [
    (101, 1, 'Cloud Migration', 'In Progress'),
    (102, 3, 'AI Chatbot', 'Completed'),
    (103, 4, 'Q3 Audit', 'In Progress'),
    (104, 5, 'Social Campaign', 'Planning'),
    (105, 10, 'Server Upgrade', 'Completed')
]

cursor.execute('DELETE FROM projects')
cursor.executemany('INSERT INTO projects VALUES (?, ?, ?, ?)', project_data)
conn.commit()

# The JOIN Query
query_join = '''
SELECT e.name, e.department, p.project_name, p.status 
FROM employees e
JOIN projects p ON e.id = p.employee_id
'''
print("Employee Projects:\n", pd.read_sql(query_join, conn), "\n")


# ==========================================
# EXERCISE 5: SQL to Pandas
# ==========================================
print("--- EXERCISE 5: Pandas Operations ---")

# Load entire table into Pandas
df = pd.read_sql("SELECT * FROM employees", conn)

# 1. Find top 3 highest paid
top_3 = df.nlargest(3, 'salary')
print("Top 3 Highest Paid:\n", top_3[['name', 'salary']], "\n")

# 2. Average salary by department using Pandas
avg_sal_pandas = df.groupby('department')['salary'].mean()
print("Average Salary by Dept (Pandas):\n", avg_sal_pandas, "\n")

# 3. Plot a bar chart
# (This will pop up in your Jupyter Notebook output)
avg_sal_pandas.plot(kind='bar', title='Average Salary by Department', color='skyblue', edgecolor='black')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.xticks(rotation=0) # Keeps the text horizontal
plt.show()

# ALWAYS close the connection when finished!
conn.close()
