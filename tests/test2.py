import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = np.load("LRoot_clean_extracted.npz")
X = data['arr_0']
y = data['arr_1']

density, cls = np.histogram(y, bins=max(y) + 1)
Xstack = np.vstack((X,y)).transpose()
Xfilter = np.array(list(filter(lambda x: density[x[1]] > 200, Xstack)))
yfilter = Xfilter[:, 1]
Xfilter = Xfilter[:, 0]
nclasses = len(set(yfilter))

Xfilter = np.array(list(map(np.array, Xfilter)))
yfilter = yfilter.astype("int")

Xtrain, Xtest, ytrain, ytest = train_test_split(Xfilter, yfilter, test_size=0.1)

logreg = linear_model.LogisticRegression()
logreg.fit(Xtrain, ytrain)

print("Train score: {}".format(logreg.score(Xtrain, ytrain)))
print("Test score: {}".format(logreg.score(Xtest, ytest)))
