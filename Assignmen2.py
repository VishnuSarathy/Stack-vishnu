# %% [markdown]
# 1. Exploratory Data Analysis (and) 2. Build a Linear Regression Model

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def sse(x,y,theta,m):
    a = 1/(2*m)
    b = np.sum(((x@theta)-y)**2)
    return a*b
def grad_dec(x,y,theta,m):
    alpha = 0.000001
    iteration = 375 #approximately found this value by manually changing and checking mse, mae. I can automate this to find optimal value but due to time constrains i was not able to..
    # his = np.zeros([iteration,1]) 
    for i in range(iteration):
        error = (x@theta) - y
        temp0 = theta[0] - (alpha/m)*np.sum(error)
        temp1 = theta[1] - (alpha/m)*np.sum(error*x[:,1])
        theta = np.array([temp0,temp1]).reshape(2,1)
    return theta

# %%
data = pd.read_csv('/Users/vishnu/Downloads/extended_salary_data.csv',sep = ',',usecols=["YearsExperience","Salary"])
data = np.array(data.values,'float')
X = data[:40,0] # 80% of the values for training 
cop = X.copy() # copying for ploting the final graph as the shape changes
Y = data[:40,1] # 80% of the values for training
# X = x/max(x)
plt.plot(X,Y,"bo")
plt.ylabel('YearsExperience')
plt.xlabel('Salary')
plt.grid()
plt.show()
m = np.size(X)
X = X.reshape(40,1)
X= np.hstack([np.ones_like(X),X])
theta = np.zeros([2,1])
# np.shape(theta)



# %% [markdown]
# 3. Evaluate the Model

# %%
theta = grad_dec(X,Y,theta,m)
plt.plot(cop,Y,"bo")
plt.plot(cop,X@theta,"-")
plt.ylabel('YearsExperience')
plt.xlabel('Salary')
plt.grid()
plt.show()
test_X = data[40:50,0] # 20% of the values for testing
A = data[40:50,1] # 20% of the values for testing
test_X.reshape(10,1)
A.reshape(10,1)
B = test_X * theta[1] + theta[0] # this is predicted output 
# C = test_X.dot(theta)
mse = ((A - B)**2).mean(axis=0)
mae = np.absolute(A-B).mean()
print("MSE: ",mse,"\nMAE: ",mae)


# %% [markdown]
# 

# %% [markdown]
# Bonus Task

# %%
NumProjectsCompleted = data[:,0]/2 + 2 #linearly corelating with year of experience 
input = np.hstack([X,NumProjectsCompleted[0:40].reshape(40,1)]) #input has format as [1,yearofexp , NumofProjects], X is from previous part
def grad_dec2(x,y,theta,m):
    alpha = 0.000001 #approximately found this value by manually changing and checking mse, mae. I can automate this to find optimal value but due to time constrains i was not able to..
    iteration = 300  #approximately found this value by manually changing and checking mse, mae. I can automate this to find optimal value but due to time constrains i was not able to..
    for i in range(iteration):
        error = (x@theta) - y
        temp0 = theta[0] - (alpha/m)*np.sum(error)
        temp1 = theta[1] - (alpha/m)*np.sum(error*x[:,1])
        temp2 = theta[2] - (alpha/m)*np.sum(error*x[:,2])
        theta = np.array([temp0,temp1,temp2]).reshape(3,1)
    return theta
theta = np.zeros([3,1])
theta = grad_dec2(input,Y,theta,40)
C = theta[0] + theta[1]*input[:,1] +theta[2]*input[:,2] #c has predicted values
ax = plt.axes(projection="3d")
ax.scatter(input[:, 1], input[:, 2], Y, label="Actual Y")
ax.set_xlabel("YearsExperience")
ax.set_zlabel("Salary")
ax.set_ylabel("ProjectsCompleted")
ax.plot(input[:, 1], input[:, 2], C, label="Predicted Y")
ax.legend()


# %% [markdown]
# Testing

# %%
A = data[40:50,1] # A has actual output
B = theta[0] + theta[1]*test_X +theta[2]*NumProjectsCompleted[40:50] # B has prredicted output
mse = ((A - B)**2).mean(axis=0)
mae = np.absolute(A-B).mean()
print(mse,mae)


