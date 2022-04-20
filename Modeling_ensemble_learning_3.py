#Boosting: 먼저 만들어진 local model의 단점을 보완하는 다음 local model

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Data
X, Y = make_moons(n_samples=600, noise=0.5, random_state=30)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=30)

#1. AdaBoost (Adaptive boosting)
result_1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=1.,
                             algorithm='SAMME.R',random_state=30)
result_1.fit(X_train, Y_train)
Y_pred_1 = result_1.predict(X_test)
print("Adaptive Boosting ensemble learning 했을 때의 정확도: ", accuracy_score(Y_test, Y_pred_1))

#2. Gradient boosting
result_2 = GradientBoostingClassifier(loss='deviance',learning_rate=0.01, n_estimators=200, random_state=30)
result_2.fit(X_train, Y_train)
Y_pred_2 = result_2.predict(X_test)
print("Gradient Boosting ensemble learning 했을 때의 정확도: ", accuracy_score(Y_test, Y_pred_2))