import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('vehicles.csv')
df = df.drop(columns=['make'])

X = df.drop(columns=['mpg'])
y = df['mpg']

#linear regression
model = LinearRegression()
model.fit(X, y)

#unscaled data point
data_point = [[6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4]]

mpg_pred = model.predict(data_point)

print(f"The predicted mpg for the given data point is: {mpg_pred[0]}")
