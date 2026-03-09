import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# #Matplotlib (Line Plot)
# days=[1,2,3,4,5,6,7,8]
# scores=[60,65,70,80,85,90,95,99]
# plt.plot(days,scores)
# plt.title("My Learning Process")
# plt.xlabel("Days")
# plt.ylabel("Scores")
# plt.show()

# #Bar chart
# Subjects=["English","Ai","ML","Math"]
# Marks=[50,60,70,80]
# plt.bar(Subjects,Marks, color = "blue")
# plt.title("Marks by subject")
# plt.xlabel("Subjects")
# plt.ylabel("Marks")
# plt.show()

# #Histogram used to see data distribution — used constantly in ML:
# scores= np.random.randint(50,100,100)
# plt.hist(scores,bins=10,color="green",edgecolor="Black")
# plt.title("Score Dirstrubtion")
# plt.xlabel("Score")
# plt.ylabel("Frequency")
# plt.show()



# # Create a blank figure
# plt.figure(figsize=(8, 4))
# plt.axis('off')  # Hide the axes/grid

# # Add your name
# plt.text(0.5, 0.5, 'RAHEEL', 
#          fontsize=80, 
#          fontweight='bold', 
#          color='#1f77b4',       # A nice professional blue
#          ha='center',           # Horizontal alignment
#          va='center',           # Vertical alignment
#          family='serif',        # Font style
#          bbox=dict(facecolor='none', edgecolor='#1f77b4', pad=10, lw=2))

# plt.show()
# #Scatter Plot
# height = [150, 160, 165, 170, 175, 180, 185]
# weight = [50, 60, 65, 70, 75, 80, 85]
# plt.scatter(height,weight,color="red")
# plt.title("Height Vs Weight")
# plt.xlabel("Height(cm)")
# plt.ylabel("Weight(kg)")
# plt.show()

# #Seaborn
# import seaborn as sns
# import pandas as pd
# data={
#     "name":["Raheel","Malik","Shaukat","Ali"],
#     "score":[90,100,80,75]
# }
# df=pd.DataFrame(data)
# sns.barplot(x="name",y="score",data=df)
# plt.title("Student Score")
# plt.show()

# #Heatmap
# data = {
#     "marks": [85, 90, 78, 92, 88],
#     "attendance": [90, 85, 70, 95, 80],
#     "assignments": [80, 95, 75, 90, 85]
# }
# df=pd.DataFrame(data)
# correlation=df.corr()
# sns.heatmap(correlation,annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.show()

# #Boxplot
# data = {
#     "subject": ["Math", "Math", "Math", "AI", "AI", "AI", "English", "English", "English"],
#     "marks": [85, 90, 78, 92, 88, 95, 70, 75, 80]
# }
# df = pd.DataFrame(data)
# sns.boxplot(x="subject",y="marks", data=df)
# plt.title("Marks distribution over the subject")
# plt.show()

# #Pairplot
# df=sns.load_dataset("iris")
# sns.pairplot(df,hue="species")
# plt.show()


#Exercise 1 
day=[1,2,3,4,5,6,7]
score=[60,65,70,75,80,85,90]
plt.plot(day,score)
plt.title("My progress")
plt.xlabel("day")
plt.ylabel("score")
plt.show()

#Exercise 2
Students=["Raheel","Malik","Shaukat","Ali","Aqeel"]
Marks=[50,60,70,80,90]
bar_colors=["red","blue","orange","green","grey"]
plt.bar(Students,Marks, color=bar_colors)
plt.title("AI/ML bar Plot")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

#Excercise 3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create the DataFrame
data = {
    'marks': [85, 45, 92, 78, 60],
    'attendance': [95, 40, 98, 80, 55],
    'assignments': [9, 3, 10, 8, 5],
    'projects': [2, 0, 3, 2, 1]
}
df = pd.DataFrame(data)

# 2. Calculate the correlation matrix
# This gives a value between -1 and 1 for every pair of columns
corr = df.corr()

# 3. Plot the Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Student Metrics")
plt.show()
#Excercise 4
import matplotlib.pyplot as plt

# Realistic data for 10 students
study_hours = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12]
exam_scores = [35, 45, 50, 65, 60, 75, 85, 80, 92, 98]

plt.scatter(study_hours, exam_scores, color="darkblue", marker="o")

# Adding a trend line (Optional but helpful)
plt.title("Study Hours vs Exam Scores")
plt.xlabel("Hours Spent Studying")
plt.ylabel("Score Achieved")
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()