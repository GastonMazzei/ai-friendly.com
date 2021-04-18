def smth_else():
    verbose = False

    df = df[df['PER'].between(0,20000)].sample(frac=1)
    print(df.head())
    x, y = df.to_numpy()[:,:-1].reshape(-1,1), df.to_numpy()[:,-1].reshape(-1,1)

    f,ax = plt.subplots(1,2)
    ax[0].hist(x)
    ax[0].set_title('x prev')
    ax[1].hist(y)
    ax[1].set_title('y prev')
    if verbose: plt.show()
    
    if False:
      for s in [StandardScaler]:
        x=s().fit_transform(x)
        y=s().fit_transform(y)

    f,ax = plt.subplots(1,2)
    ax[0].hist(x)
    ax[0].set_title('x post')
    ax[1].hist(y)
    ax[1].set_title('y post')
    if verbose: plt.show()
    else: plt.close()

    if False:
      answ = input('please insert the radius range: e.g. (x y)')
      rmin = float(answ.split(' ')[0])
      rmax = float(answ.split(' ')[1])
      y = np.asarray([[1] if rmin<x[0]<rmax else [0] for x in y])  
    elif False:
      y = Binarizer(threshold=np.mean(y)).fit_transform(y)    

    L = df.shape[0]
    divider = {'train':slice(0,int(0.7*L)),
               'val':slice(int(0.7*L),int((0.7+0.15)*L)),
               'test':slice(-int(0.15*L),None),}

    for k,i in divider.items():
        data[k] = (x[i],y[i])
        print(f'for key {k} {np.count_nonzero(data[k][1])/len(data[k][1])*100}% are non-zero')

    answ = input('if you are happy with the ratio, press "y"... else "n"')
    if answ=='y': return data
    else: return preprocess(df)

def manage_database(name):
  with open(name,'rb') as f:
    data = pickle.load(f)
  return




