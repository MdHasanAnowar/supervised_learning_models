import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.2)

contents = pd.read_csv('iris.data') # reading iris data
data = contents[49:149] # Keeping only Iris-versicolor and Iris-virginica data

sepal_length = data.iloc[:, 0] # selecting sepal length of Iris-versicolor and Iris-virginica
petal_width = data.iloc[:, 3] # selecting petal width of Iris-versicolor and Iris-virginica

def linear_regression_gd (x, y, max_iter, tol, learning_rate):
    theta = [0, 0] # initialize theta values
    iters = 0  # initialize iteration count
    n = len(x)  # number of elements in independent variable

    h_theta = theta[0] + theta[1] * x  # linear regression equation
    j_theta = (1 / n) * sum((h_theta - y) ** 2)  # cost function 𝐽(𝜃)
    all_cost = [j_theta]  # values of cost function
    diff_j_theta = all_cost[0]

    # Gradient Descent
    while iters != max_iter and diff_j_theta > tol:
        der_theta1 = (2 / n) * sum(x * (h_theta - y))  # Derivative of cost function wrt coefficient theta[1]
        der_theta0 = (2 / n) * sum(h_theta - y)  # Derivative of cost function wrt constant theta[0]
        theta[1] -= learning_rate * der_theta1  # Update theta[1]
        theta[0] -= learning_rate * der_theta0  # Update theta[0]\
        h_theta = theta[0] + theta[1] * x # updated linear regression equation
        j_theta = (1 / n) * sum((h_theta - y) ** 2) # updated cost function 𝐽(𝜃)
        all_cost.append(j_theta)
        diff_j_theta = all_cost[iters] - all_cost[iters + 1]
        iters += 1

    return theta, all_cost, iters

max_iter = 50000  # maximum number of iterations
tol = 0.0000001 # tolerance

# part 1b
# selecting sepal length as independent and  petal width as dependent variables

# learning_rate = 0.00001
[theta, all_cost, iters] = linear_regression_gd (x = sepal_length, y = petal_width,
                                                 max_iter = max_iter, tol = tol, learning_rate = 0.00001)
print("Values of theta = ",theta)
print("Number of iterations = ",iters)
print("Initial cost value = ", all_cost[0])
print("Final cost value = ", all_cost[iters])

# Regression Line Plot
plt.figure()
plt.scatter(sepal_length, petal_width, color = 'green')
h_theta = theta[0] + theta[1] * sepal_length
plt.plot(sepal_length, h_theta, color = 'indianred') # plotting the linear regression line
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Linear Regression Using Gradient Descent\n(learning_rate = 0.00001)')
plt.show()

# Convergence plot
plt.plot(range(iters+1),all_cost)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')
plt.title('Convergence plot of Petal Width Vs. Sepal Length\n(learning_rate = 0.00001)')
plt.show()


# learning_rate = 0.000001
[theta, all_cost, iters] = linear_regression_gd (x = sepal_length, y = petal_width,
                                                 max_iter = max_iter, tol = tol, learning_rate = 0.000001)
print("\nValues of theta = ", theta)
print("Number of iterations = ", iters)
print("Initial cost value = ", all_cost[0])
print("Final cost value = ", all_cost[iters])

# Regression Line Plot
plt.scatter(sepal_length, petal_width, color = 'green')
h_theta = theta[0] + theta[1] * sepal_length
plt.plot(sepal_length, h_theta, color = 'indianred') # plotting the linear regression line
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Linear Regression Using Gradient Descent\n(learning_rate = 0.000001)')
plt.show()

# Convergence plot
plt.plot(range(iters+1),all_cost)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')
plt.title('Convergence plot of Petal Width Vs. Sepal Length\n(learning_rate = 0.000001)')
plt.show()


# selecting petal width as independent and sepal length as dependent variables
# Learning_rate = 0.00001
max_iter = 400000
tol = 0.0000001
[theta, all_cost, iters] = linear_regression_gd (x = petal_width, y = sepal_length,
                                                 max_iter = max_iter, tol = tol, learning_rate = 0.00001)
print("\nValues of theta = ",theta)
print("Number of iterations = ",iters)
print("Initial cost value = ", all_cost[0])
print("Final cost value = ", all_cost[iters])

# Regression Line Plot
plt.scatter(petal_width, sepal_length, color = 'green')
h_theta = theta[0] + theta[1] * petal_width
plt.plot(petal_width, h_theta, color = 'indianred') # plotting the linear regression line
plt.xlabel('Petal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Linear Regression Using Gradient Descent\n(learning_rate = 0.00001)')
plt.show()

# Convergence plot
plt.plot(range(iters+1),all_cost)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')
plt.title('Convergence plot of Sepal Length Vs. Petal Width\n(learning_rate = 0.00001)')
plt.show()


# Learning_rate = 0.000001
[theta, all_cost, iters] = linear_regression_gd (x = petal_width, y = sepal_length,
                                                 max_iter = max_iter, tol = tol, learning_rate = 0.000001)
print("\nValues of theta = ", theta)
print("Number of iterations = ", iters)
print("Initial cost value = ", all_cost[0])
print("Final cost value = ", all_cost[iters])

# Regression Line Plot
plt.scatter(petal_width, sepal_length, color = 'green')
h_theta = theta[0] + theta[1] * petal_width
plt.plot(petal_width, h_theta, color = 'indianred') # plotting the linear regression line
plt.xlabel('Petal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Linear Regression Using Gradient Descent\n(learning_rate = 0.000001)')
plt.show()

# Convergence plot
plt.plot(range(iters+1),all_cost)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')
plt.title('Convergence plot of Sepal Length Vs. Petal Width\n(learning_rate = 0.000001)')
plt.show()
