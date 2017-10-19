import numpy as np
import pylab as pl
import matplotlib.pyplot as pl

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import maxabs_scale, FunctionTransformer, MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from inspect import getmembers

data=np.loadtxt(open("data_one.csv","rb"),delimiter=",",skiprows=0)

enc = OneHotEncoder()
enc.fit(data)

print(enc.n_values_)
print(enc.feature_indices_)
print(enc.active_features_)

Tdata = OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
TT=Tdata.fit_transform(data).toarray()

print TT[0]

X, y =TT[:,:76],TT[:,77]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

et=ExtraTreesClassifier(n_estimators=500,verbose=1,max_depth=5,criterion='entropy')
et.fit(X_train,y_train)
importance=et.feature_importances_

print(importance)

#print(et.decision_path(X_train))
#input("stop")
print(et.score(X_test,y_test))




rf=RandomForestClassifier(criterion='entropy',n_estimators=500,random_state=0,n_jobs=2,max_depth=6,max_features='log2')
rf.fit(X_train,y_train)

print(rf.score(X_test,y_test))


kn=KNeighborsClassifier(n_neighbors=3, p=1, metric='minkowski')
kn.fit(X_train,y_train)

print(kn.score(X_test,y_test))

bb= BernoulliNB(alpha=0.8)
bb.fit(X_train,y_train)

print(bb.score(X_test,y_test))


gn=GaussianNB()
gn.fit(X_train,y_train)

print(gn.score(X_test,y_test))

mlp = MLPClassifier(activation='relu',solver='sgd',learning_rate='adaptive', learning_rate_init=0.001, max_iter=2000, hidden_layer_sizes=(50,100,100), random_state=None)
mlp.fit(X_train,y_train)

print(mlp.score(X_test,y_test))
