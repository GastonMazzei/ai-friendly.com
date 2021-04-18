from datasets_generator import *

#df = main(20000, False,500)

df = main(100000, True,100,50)

print(df.head)

df.to_csv('database/database.csv', index=False)
