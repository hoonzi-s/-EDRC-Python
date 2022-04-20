#cross validation example (단순 분리)

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

boston = load_boston()
dfX = pd.DataFrame(boston.data, columns=boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=["MEDV"])
df = pd.concat([dfX, dfy], axis=1) #합치기

#전체 데이터 중에서 80%만 뽑아서 train 시키고 나머지 20%는 검증에 사용하자.
N = len(df)
ratio = 0.8
np.random.seed(0)
idx_train = np.random.choice(np.arange(N), np.int(ratio * N))
idx_test = list(set(np.arange(N)).difference(idx_train))

df_train = df.iloc[idx_train]
df_test = df.iloc[idx_test]

model = sm.OLS.from_formula("MEDV ~ " + "+".join(boston.feature_names), data=df_train)
result = model.fit()
print(result.summary())

#단순 데이터 분리 (학습용/검증용)
df_train_new, df_test_new = train_test_split(df, test_size=0.2, random_state=0)
print(df_train_new.shape)
print(df_test_new.shape)