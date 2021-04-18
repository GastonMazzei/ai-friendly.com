import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv


color = {'true value':'r', 'predicted value':'b'}
df = pd.read_csv('database/network-regression-predictions.csv')

fig = plt.figure(figsize=plt.figaspect(0.5))

for i,n in enumerate(['true value','predicted value']):
  ax = fig.add_subplot(1, 2, i+1, projection='3d')
  df2 = df[df['type']=='true value'].sort_values(by=['radius','mass','period'])
  x,y,z = [df2[q].to_numpy().tolist() for q in ['mass', 'radius','period']]
  ax.plot(x, y, z, color=color[n], label=n, linewidth=3.5)#cmap=cm.jet, linewidth=0.1)
  ax.legend()
  ax.set_xlabel('mass')
  ax.set_ylabel('radius')
  ax.set_zlabel('period')
plt.savefig('gallery/regression-testing-predictions.png')


