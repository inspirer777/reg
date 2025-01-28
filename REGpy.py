#https://datatofish.com/multiple-linear-regression-python/
from sklearn import linear_model
import pandas as pd
import statsmodels.api as sm

path_to_file = 'C:/Users/BEHINLAPTOP/Desktop/D.csv'
df = pd.read_csv(path_to_file)
df.head()
df.shape
x = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
y = df['y']


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)
######################################
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection ='3d')
 
ax.scatter(df['x1'], df['x2'], df['y'], label ='y', s = 5)
ax.legend()
ax.view_init(45, 0)
 
plt.show()

import statsmodels.formula.api as smf
linear_regression = smf.ols(formula = 'y ~ x1 + x2 +x3 + x4 + x5 + x6 + x6 ', data=df)
fitted_model = linear_regression.fit()
fitted_model.summary()
############################################

#https://www.educative.io/answers/how-to-implement-multivariable-regression-in-python
#enerating training and testing data from our data:
# We are using 80% data for training.
train = df[:(int((len(df)*0.8)))]
test = df[(int((len(df)*0.8))):]

# Modeling:
# Using sklearn package to model data :
regr = linear_model.LinearRegression()
train_x = np.array(train[["x1"]])
train_y = np.array(train[["y"]])
regr.fit(train_x,train_y)

ax.scatter(df["x1"],df["y"],df["x2"],color="red")
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
ax.set_xlabel("x1")
ax.set_ylabel("y")
plt.show()
print ("coefficients : ",regr.coef_) #Slope
print ("Intercept : ",regr.intercept_)


# Predicting values:
# Function for predicting future values
def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values

# Checking various accuracy:
from sklearn.metrics import r2_score 
test_x = np.array(test[['x1']]) 
test_y = np.array(test[['y']]) 
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y)** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )










