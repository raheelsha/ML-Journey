# list comprehension in the python
numbers= [10,20,30,40]
squared= [n**2 for n in numbers ]
print(squared)

# Double every number
number=[2,3,4,5]
doubled=[n*2 for n in number]
print(doubled)

#uppercase every word
cities=["Lahore","karachi","faislabad","chakwal"]
upper_cities=[city.upper() for city in cities]
print(upper_cities)

#filter even number
Number=[1,2,3,4,5,6,7,8,9,10]
even_number=[n for n in Number if n%2==0]
print(even_number)

#Numpy array
import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9])
doubled=arr*2
print(doubled)
print(type(arr))

ones=np.ones(100)
print(ones)

#Random Numbers
random_arr=np.random.rand(5)
print(random_arr)

#Array Operations

arr=np.array([3,4,5,6,10,20,100])
print(arr*2)
print(arr+2)
print(arr**2)

#Statistics on the array
print(np.mean(arr))
print(np.median(arr))
print(np.max(arr))
print(np.min(arr))
print(np.sum(arr))
print(np.std(arr))


#list of square and print those which are greater than 100
arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

squares=arr**2
result=squares[squares>100]
print(result)

#A NumPy array of 10 random numbers. Then print the mean, max, min, and standard deviation of the array.
random_arr=np.random.rand(10)
print(random_arr)
print(np.mean(random_arr))
print(np.max(random_arr))
print(np.min(random_arr))
print(np.std(random_arr))

#Create a 2D NumPy array that represents this student marks table
marks = np.array([
    [85, 90, 95],
    [70, 75, 80],
    [60, 65, 68]
])
print("Student 2 marks:", marks[1])
print("AI marks for all students:", marks[:, 2])
class_average = np.mean(marks)
print(f"The average marks of the entire class: {class_average:.2f}")




