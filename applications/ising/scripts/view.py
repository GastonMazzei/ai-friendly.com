import sys

import pandas as pd
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

df=pd.read_csv('dataset/ising.csv')
try: 
  show = int(sys.argv[1])
except:
  show = 0

try: 
  index = int(sys.argv[2])
except:
  index = 0
variable = ['M','E','X','C'][index]
fancy_name = {'M':'Magnetization','E':'Energy','X':'Susceptibility',
              'C':'Capacity'}
if show: print(f'Showing results for {fancy_name[variable]}')

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df['T'],df.field,df[variable], c=df[variable])
ax.set_xticks([0,1,2,3,4])
ax.set_yticks([-1,0,1])
zmin, zmax = min(df[variable]), max(df[variable])
zmiddle = zmin + (zmax-zmin)/2
ax.set_zticks([zmin,zmiddle,zmax])
ax.set_xlabel('Temperature')
ax.set_ylabel('Field Intensity')
ax.set_zlabel(f'System\'s {fancy_name[variable]}')
plt.title(f'3dplot')
plt.savefig(f'gallery/3dplot-{fancy_name[variable]}.png')
if show: plt.show()

f, ax = plt.subplots()
plt.scatter(df['T'],df.field, c=df[variable])
ax.set_xticks([0,1,2,3,4])
ax.set_yticks([-1,0,1])
ax.set_xlabel('Temperature')
ax.set_ylabel('Field Intensity')
ax.set_title(f'2dplot')
clb = plt.colorbar()
clb.ax.set_title(f'System\'s {fancy_name[variable]}')
f.savefig(f'gallery/2dplot-{fancy_name[variable]}.png')
if show: plt.show()
