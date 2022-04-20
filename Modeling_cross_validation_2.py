#cross validation example (N-fold)

from sklearn.model_selection import KFold
import statsmodels.api as sm
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

#Data
boston = load_boston()
dfX = pd.DataFrame(boston.data, columns=boston.feature_names) #x
dfy = pd.DataFrame(boston.target, columns=["MEDV"]) #y
df = pd.concat([dfX, dfy], axis=1) #합치기

#K-fold split
scores = []
cv = KFold(10, shuffle=True, random_state=0) #객체 생성

for i, (idx_train, idx_test) in enumerate(cv.split(df)):
    df_train = df.iloc[idx_train]
    df_test = df.iloc[idx_test]

    model = sm.OLS.from_formula("MEDV ~ " + "+".join(boston.feature_names), data=df_train) #model
    result = model.fit()

    pred = result.predict(df_test) #prediction
    rss = ((df_test.MEDV - pred) ** 2).sum() #각각의 square error
    tss = ((df_test.MEDV - df_test.MEDV.mean()) ** 2).sum() #total square error
    rsquared = 1 - rss / tss

    scores = np.append(scores, rsquared)
    print("학습 R sqaure 값 = {:.8f}, 검증 R square 값 = {:.8f}".format(result.rsquared, rsquared))