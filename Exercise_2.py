from pyexpat import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

df = pd.read_csv('vehicles.csv')

df = df.drop(columns=['make'])

# Define the target variable and features
features = ['wt', 'am', 'qsec', 'drat', 'gear']
target = 'mpg'

#meshgrid for the features
wt, am, qsec, drat, gear = np.meshgrid(
    np.linspace(df['wt'].min(), df['wt'].max(), 10),
    np.linspace(df['am'].min(), df['am'].max(), 10),
    np.linspace(df['qsec'].min(), df['qsec'].max(), 10),
    np.linspace(df['drat'].min(), df['drat'].max(), 10),
    np.linspace(df['gear'].min(), df['gear'].max(), 10)
)

# Flatten the meshgrid arrays
wt_flat = wt.flatten()
am_flat = am.flatten()
qsec_flat = qsec.flatten()
drat_flat = drat.flatten()
gear_flat = gear.flatten()

#DataFrame from the flattened meshgrid arrays
meshgrid_df = pd.DataFrame({
    'wt': wt_flat,
    'am': am_flat,
    'qsec': qsec_flat,
    'drat': drat_flat,
    'gear': gear_flat
})

#Predict mpg values using the linear regression 
model = LinearRegression()
model.fit(df[features], df[target])
mpg_pred = model.predict(meshgrid_df)

#6D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(wt_flat, am_flat, qsec_flat, c=mpg_pred, cmap='viridis')
plt.colorbar(sc, label='mpg')
ax.set_xlabel('wt')
ax.set_ylabel('am')
ax.set_zlabel('qsec')
plt.title('6D Plot with mpg assigned to marker color')

plt.show()
