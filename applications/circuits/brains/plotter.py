import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def view_database():
  df = pd.read_csv("database/database.csv").drop(['D'],axis=1)
  sns.stripplot(data=df)
  plt.yscale('log')
  plt.savefig('gallery/database-parameters')
  return
