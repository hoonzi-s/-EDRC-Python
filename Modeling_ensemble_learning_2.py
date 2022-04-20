#Bagging (Bootstrap aggregating): 중복을 허용한 랜덤 추출

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
import pandas as pd

#Data
X, Y = make_moons(n_samples=600, noise=0.5, random_state=30)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=30)

#local model 하나만 사용했을 때
result_1 = DecisionTreeClassifier(random_state=30)
result_1.fit(X_train, Y_train)
Y_pred_1 = result_1.predict(X_test)
print("하나의 local model만 사용했을 때의 정확도: ", accuracy_score(Y_test, Y_pred_1))

#local model을 여러개 사용했을 때 (Ensemble)
result_2 = BaggingClassifier(DecisionTreeClassifier(random_state=30), n_estimators=100, max_samples=100,
                             bootstrap=True, n_jobs=-1, random_state=30)
result_2.fit(X_train, Y_train)
Y_pred_2 = result_2.predict(X_test)
print("여러개의 local model을 사용하여 ensemble learning 했을 때의 정확도: ", accuracy_score(Y_test, Y_pred_2))