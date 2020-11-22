import plotly.express as px
import pandas as pd
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# We first identify the dimentionality of the dataset and the variables being measured
print(" ----- MEASURED VARIABLES ----- ")
print(iris.feature_names)

# Next, we standardize the data by...
# 1. Calculating the sample mean of each variable
sepal_length = 0
sepal_width = 0
petal_length = 0
petal_width = 0
for row in iris.data:
    for col in range(0, 4):
        if(col == 0):
            sepal_length += row[col]
        elif(col == 1):
            sepal_width += row[col]
        elif(col == 2):
            petal_length += row[col]
        else:
            petal_width += row[col]

sepal_length_mu = sepal_length/len(iris.data)
sepal_width_mu = sepal_width/len(iris.data)
petal_length_mu = petal_length/len(iris.data)
petal_width_mu = petal_width/len(iris.data)

print(" ----- SAMPLE MEANS ----- ")
print("Average sepal length: " + str(sepal_length_mu))
print("Average sepal width: " + str(sepal_width_mu))
print("Average petal length: " + str(petal_length_mu))
print("Average petal width: " + str(petal_width_mu))

# 2. Calculating the sample standard deviation of the variable
sepal_length_sigma = 0
sepal_width_sigma = 0
petal_length_sigma = 0
petal_width_sigma = 0

for row in iris.data:
    for col in range(0, 4):
        if(col == 0):
            sepal_length_sigma += (row[col] - sepal_length_mu) ** 2
        elif(col == 1):
            sepal_width_sigma += (row[col] - sepal_width_mu) ** 2
        elif(col == 2):
            petal_length_sigma += (row[col] - petal_length_mu) ** 2
        else:
            petal_width_sigma += (row[col] - petal_width_mu) ** 2

sepal_length_sigma **= 1/2
sepal_length_sigma /= len(iris.data) - 1
sepal_width_sigma **= 1/2
sepal_width_sigma /= len(iris.data) - 1
petal_length_sigma **= 1/2
petal_length_sigma /= len(iris.data) - 1
petal_width_sigma **= 1/2
petal_width_sigma /= len(iris.data) - 1

print(" ----- SAMPLE STANDARD DEVIATIONS ----- ")

print("Sepal length standard deviation: " + str(sepal_length_sigma))
print("Sepal width standard deviation: " + str(sepal_width_sigma))
print("Petal length standard deviation: " + str(petal_length_sigma))
print("Petal width standard deviation: " + str(petal_width_sigma))

# 3. For every observation, subtract its sample mean and divide by its sample standard deviation
for row in iris.data:
    for col in range(0, 4):
        if(col == 0):
            row[col] -= sepal_length_mu
            row[col] /= sepal_length_sigma
        elif(col == 1):
            row[col] -= sepal_width_mu
            row[col] /= sepal_width_sigma
        elif(col == 2):
            row[col] -= petal_length_mu
            row[col] /= petal_length_sigma
        else:
            row[col] -= petal_width_mu
            row[col] /= petal_width_sigma

# Calculate the covariance matrix
print(" ----- COVARIANCE MATRIX ----- ")
covariance_matrix = np.matmul(iris.data.transpose(), iris.data)
covariance_matrix /= len(iris.data) - 1
print(covariance_matrix)

# Calculate eigenvalues and eigenvectors
print(" ----- EIGENVALUES AND EIGENVECTORS ----- ")
e_vals, e_vecs = np.linalg.eig(covariance_matrix)
print(e_vals)
print(e_vecs)

# Calculate the trace of the matrix and the proportion of total variance accounted for by each principal component
trace = 0
for row in range(0, len(covariance_matrix)):
    for col in range(0, len(covariance_matrix[row])):
        if(row == col):
            trace += covariance_matrix[row][col]

pc1_var_prop = e_vals[0]/trace
pc2_var_prop = e_vals[1]/trace
pc3_var_prop = e_vals[2]/trace
pc4_var_prop = e_vals[3]/trace

print(" ----- PROPORTION OF VARIANCE ----- \n")
print("PC1: " + str(pc1_var_prop))
print("PC2: " + str(pc2_var_prop))
print("PC3: " + str(pc3_var_prop))
print("PC4: " + str(pc4_var_prop))

# Create the projection matrix using first two principal components
proj_mat = np.matmul(iris.data, e_vecs.transpose()[:2].transpose())

print(" ----- PROJECTION MATRIX -----")

proj_mat = proj_mat.tolist()

for row in range(0, 150):
    proj_mat[row].append(iris.target_names[iris.target[row]])

df = pd.DataFrame(proj_mat, iris.target, columns=[
                  'Principal Component 1', 'Principal Component 2', 'Iris Species'])

print(df)

# Plot the data (opens in browser)
fig = px.scatter(df, x="Principal Component 1",
                 y="Principal Component 2", color="Iris Species")

fig.update_traces(marker=dict(size=12))
fig.show()
