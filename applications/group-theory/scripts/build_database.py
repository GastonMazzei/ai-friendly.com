import re
import os
import numpy as np
import pandas as pd


# Make the "Y"
with open('raw/lenvecs.txt','r') as f:
  y = f.readlines()
pattern1 = re.compile(f'([\-0-9]+)')
y = re.findall(pattern1,y[0])
y = np.asarray([int(x_) for x_ in y])

# Make the "X"
with open('raw/padvecs.txt','r') as f:
  x = f.readlines()

if False:
  pattern2 = re.compile('\[([\-0-9,]+)\]')
  x = re.findall(pattern2,x[0])
  x = [[int(k) for k in re.findall(pattern1,x_)] for x_ in x]
else:
  x = re.findall(pattern1,x[0])
  x = [int(x_) for x_ in x]  
x = np.asarray(x).reshape(-1,8)

# Print both's shapes
print(x.shape, y.shape)

# Create a pandas.DataFrame
df=pd.DataFrame(np.concatenate([x,y.reshape(-1,1)],1))

# Show different thresholds!
if False:
  import matplotlib.pyplot as plt
  f, ax = plt.subplots(1,2)
  df[8].apply(lambda x: 1 if x > 9 else 0).hist(ax=ax[0])
  df[8].apply(lambda x: 1 if x >= 9 else 0).hist(ax=ax[1])
  ax[0].set_title('x>9')
  ax[1].set_title('x>=9')
  plt.show()

# Binarize the dataset for binary classification 
#and save it as "database/database.csv"
df[8] = df[8].apply(lambda x: 1 if x>9 else 0)
try: os.mkdir('database')
except: pass
df.to_csv('database/database.csv',index=False)
print('SAVED SUCCESSFULLY')


