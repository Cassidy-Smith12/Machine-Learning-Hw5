import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('vehicles.csv')
df = df.drop(columns=['make'])

X = df.drop(columns=['mpg'])
y = df['mpg']

model = LinearRegression()
model.fit(X, y)

#coefficients
coefficients = model.coef_
feature_names = X.columns

#data frame
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# five most important weighted coefficients
top_5_features = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index).head(5)

print("The five most important weighted coefficients and their names are:")
print(top_5_features)
