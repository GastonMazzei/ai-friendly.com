import numpy as np

def rotate_to_bottom_left_center_of_mass(a):
  versions = {x:a.copy() for x in range(4)}
  index, max_value = 0, -99999 
  L = len(a)
  calc = {}
  for x in versions.keys():
    for j in range(x):
      versions[x] = np.rot90(versions[x])
    temp = 0 
    for i in range(L//2+1 if L%2==1 else L//2):
      for j in range(L//2+1 if L%2==1 else L//2):
        if versions[x][L-1-i,j] == -1: pass
        else: temp += 1 
    calc[x] = temp 
    new_max_value = max(max_value, calc[x])
    if new_max_value > max_value: 
      index = x
      max_value = new_max_value
  return versions[index],index, int((4-index)%4)      

