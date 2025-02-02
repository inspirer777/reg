# Multiple Linear Regression for Building Energy Analysis

This project is designed to predict heating and cooling loads in buildings based on various structural features of the buildings. The data is derived from simulations performed in the Ecotect software and includes various features such as surface area, overall height, and glazing area. The aim is to predict the heating load and cooling load of buildings using these features.

## Dataset Information

We perform energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ in terms of glazing area, glazing distribution, orientation, and other parameters. Various settings are simulated as functions of these characteristics to generate 768 building shapes. The dataset consists of 768 samples and 8 features, and the goal is to predict two responses from these 8 features.

### Feature Information
The dataset contains 8 attributes (or features) and 2 responses (or outcomes). The goal is to use the 8 features to predict each of the two responses.

- **X1**: Relative Compactness
- **X2**: Surface Area
- **X3**: Wall Area
- **X4**: Roof Area
- **X5**: Overall Height
- **X6**: Orientation
- **X7**: Glazing Area
- **X8**: Glazing Area Distribution
- **y1**: Heating Load
- **y2**: Cooling Load

## Implementation Steps

In this project, two different methods for implementing multiple linear regression are used: one using the `sklearn` library and another using `statsmodels`.

### 1. Loading the Data

First, the data is loaded from a CSV file:

```python
import pandas as pd
df = pd.read_csv('C:/Users/BEHINLAPTOP/Desktop/D.csv')
df.head()
```
2. Defining Features and Responses
In this step, features and responses are separated from the data. For predicting either of the heating load or cooling load, different features are used:

```python
x = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
y = df[['y1', 'y2']]

```
3. Linear Regression using sklearn
In this part, the linear regression model is created using sklearn, and the results are printed:
```python
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x, y['y1'])  # Predicting Heating Load (y1)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

```
4. Linear Regression using statsmodels
A similar model is created using statsmodels, which provides more detailed statistical results:

```python
import statsmodels.api as sm
x = sm.add_constant(x)  # Adding a constant to the model
model = sm.OLS(y['y1'], x).fit()  # Linear regression model for predicting Heating Load (y1)
print(model.summary())

```
3D Data Visualization
For data visualization and result representation, 3D plots are used:
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['x1'], df['x2'], df['y1'], label='y1', s=5)
ax.legend()
ax.view_init(45, 0)
plt.show()

```
6. Splitting the Data into Training and Testing
In this section, the data is split into training and testing sets (80% for training and 20% for testing):

```python
train = df[:int(len(df)*0.8)]
test = df[int(len(df)*0.8):]

```
7. Model Evaluation
The created models are evaluated using the test data. Different metrics such as R-squared and Mean Absolute Error are used:

```python
from sklearn.metrics import r2_score
test_x = np.array(test[['x1']])
test_y = np.array(test[['y1']])
test_y_pred = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_pred - test_y)))
print("R2-score: %.2f" % r2_score(test_y_pred, test_y))

```
Prerequisites
To run this code, you will need the following libraries:

- numpy
- pandas
- scikit-learn
- statsmodels
- matplotlib

You can install these libraries using the following command:
```bash
pip install numpy pandas scikit-learn statsmodels matplotlib

```
Using the Code
1.First, prepare the data in CSV format.
2.Correctly set the file path in the code.
3.Then, run the code to build linear regression models to predict heating and cooling loads.


This `README.md` file provides a clear overview of the project, including the dataset, the implementation steps, and instructions for using the code.

## License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this project under the terms of the license.

Good luck !

