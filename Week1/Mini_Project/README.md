# 🚢 Titanic Dataset Analysis

## 📌 Project Overview
This project performs exploratory data analysis (EDA) on the famous Titanic dataset. 
The goal is to uncover patterns and insights about passenger survival using Python, Pandas, and data visualization libraries.

## 🛠️ Tools & Libraries Used
- **Python** — core programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computations
- **Matplotlib** — data visualization
- **Seaborn** — statistical data visualization

## 📊 Dataset
- **Source:** Seaborn built-in Titanic dataset
- **Size:** 891 passengers, 15 features
- **Features:** Survived, Pclass, Sex, Age, Fare, Embarked, and more

## 🔍 Key Steps
1. **Data Loading** — loaded dataset using Seaborn
2. **Data Exploration** — analyzed shape, info, and statistics
3. **Data Cleaning** — handled missing values in Age, Embarked, and Deck columns
4. **Visualization** — created 6 insightful plots
5. **Insights** — drew conclusions from the analysis

## 📈 Visualizations
![Titanic Analysis](titanic_analysis.png)

## 💡 Key Findings
- Overall survival rate was only **38.4%**
- **Women had 74% survival rate** vs only 19% for men
- **1st class passengers** had 63% survival rate vs 24% for 3rd class
- **Higher fare** correlated with better survival chances
- Most passengers were between **20-35 years old**
- Deck column had **77% missing values** and was dropped

## 🧹 Data Cleaning Summary
| Column | Issue | Solution |
|--------|-------|----------|
| Age | 177 missing values | Filled with median age |
| Embarked | 2 missing values | Filled with most common port |
| Deck | 688 missing values (77%) | Dropped column |

## 🏁 Conclusions
- Gender was the strongest predictor of survival
- Passenger class (wealth) significantly affected survival chances
- The Titanic disaster was not random — social factors played a huge role

## 👤 Author
**Raheel Shaukat**  
AI/ML Intern @ NetSol Technologies  
[GitHub](https://github.com/raheelsha)
