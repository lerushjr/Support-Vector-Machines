import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

iris = sns.load_dataset('iris')
sns.pairplot(iris,hue='species')
setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],data=setosa,cmap="plasma", shade=True, shade_lowest=False)
X=iris.drop('species', axis=1)
y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3)
model = SVC()
model.fit(X_train,y_train)
pred= model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))

#Grid Search to find optimal parameters
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001]}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
