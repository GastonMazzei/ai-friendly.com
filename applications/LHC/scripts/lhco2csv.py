import re
import pandas as pd
import sys

def convert(v):
  pattern = re.compile('[\s]*([a-z\.\-0-9/#]+)',re.I)
  return re.findall(pattern, v)

names = sys.argv[1]
f = open(f'raw/{names}_LHCO','r')
h = []
#limit = 50
#i = 0
for line in f:
  h += [line]
  #i += 1
  #if i>limit: break

f.close()
data = []
data_aux = []
for line in h:
  temp = convert(line)
  if temp[0]=='0': 
    # Flush
    data += [data_aux]
    data_aux = []
  else:
    data_aux += temp

M = int(max([len(x) for x in data])/len(data[0]))
#M = 12
cols = []
for x in range(M):
  cols += [y+f'_{x+1}' for y in data[0]]


df = pd.DataFrame(data[1:], columns=cols)

# Tidy Up
# .A) Add numbers
for x in range(M):
  df[f'#_{x+1}'] = x+1
# .B) replace nans with zeros
df = df.fillna(0)

# View!
if False:
  print(df.columns)
  #for x in h:
  #  print(x)

# Save!
if True:
  df.to_csv(f'database/{names}.csv',index=False)
print('len was ',len(df))
print('column length was ',len(df.columns))



	






