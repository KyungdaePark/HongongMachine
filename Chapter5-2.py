# Cross Validate
# 훈련 세트의 20% 만큼 검증 세트로 나누기
# 훈련 64% 검증 16% 테스트 20%

import numpy as np
from ast import Starred
import pandas as pd
wine = pd.read_csv('http://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine[['class']].to_numpy()

# split
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state = 42
)

# one more split
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size = 0.2, random_state = 42
)

print("SUB, VAL, TEST SIZE : ",end="")
print(sub_input.shape, val_input.shape, test_input.shape)

# dt
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(sub_input, sub_target)
print("\nDecisionTree")
print("SUB SCORE : " +str(dt.score(sub_input, sub_target)))
print("VAL SCORE : " +str(dt.score(val_input, val_target)))

# cv, k-fold cv
from sklearn.model_selection import cross_validate
score1 = cross_validate(dt, train_input, train_target)
print("\nDecisionTree, CV")
print("SCORE : " +str(score1['test_score']))
print("MEAN SCORE : "+str(np.mean(score1['test_score'])))
# 기본 5-fold cv
# VAL이 계속 바뀌며 5번에 걸쳐 검증하므로 5개의 결과값이 나온다.

# 근데 얘는 훈련 세트에서 shuffle하지 않음
# 그리고 얘는 아래 코드랑 같음

from sklearn.model_selection import StratifiedKFold
score2 = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
print("\nDecisionTree, CV (StratifiedKFold)")
print("SCORE : " + str(score2['test_score']))
print("MEAN SCORE : "+str(np.mean(score2['test_score'])))

# shuffle, 10-fold?
score3 = cross_validate(dt, train_input, train_target,
            cv = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 42)
        )
print("\nDecisionTree, CV (StratifiedKFold, 10 Fold, shuffle)")
print("SCORE : " +str(score3['test_score']))
print("MEAN SCORE : "+str(np.mean(score3['test_score'])))


# HyperParameterTuning
# GridSearchCV : 매개변수의 모든 조합을 테스트함
# cross_validate()를 사용해 CV를 굳이 만들 필요 없음.
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)

best_dt = gs.best_estimator_
print("\nGridSearchCV : 'min_impurity_decrease'")
print("BEST MODEL : " + str(best_dt))
print("BEST MODEL SCORE : " + str(best_dt.score(train_input, train_target)))

print("\n각 매개변수에서 수행한 교차검증의 평균 (0.0001에서의 평균, 0.0002에서의 평균...)")
print("EACH MEAN SCORE : " + str(gs.cv_results_['mean_test_score']))

# argmax()는 가장 큰 값의 인덱스 추출 가능
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print("\n Params of Best Index (np.argmax)")
print(gs.cv_results_['params'][best_index]) # params는 매개변수 조합 有

# 이제는 3개 이상의 조합으로 해보자.
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5,20,1),
          'min_samples_split' : range(2,100,10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print("\n6750개의 테스트 : ")
print("BEST Params : " +str(gs.best_params_))
print("각 매개변수조합에서 실행한 교차검증의 평균들의 최댓값")
print("MAX : " + str(np.max(gs.cv_results_['mean_test_score'])))

# 0.0001, 0.0002...의 스케일은 임의로 정했음.
# 이 간격을 적당히 랜덤으로 정하고. 좀 일정 범위 안에서는 골고루 뽑았으면 좋겠음.
# uniform(실수), randint(정수)
from scipy.stats import uniform, randint
params = {'min_impurity_decrease' : uniform(0.0001, 0.001),
          'max_depth' : randint(20,50),
          'min_samples_split' : randint(2,25),
          'min_samples_leaf' : randint(1,25)
          }
# 그리고 RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
gs2 = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42), 
                         params, n_iter = 100, n_jobs = -1, random_state = 42)
gs2.fit(train_input, train_target)

print("\nBEST Params with RandomizedSearchCV")
print("BEST Model : " + str(gs2.best_params_))
print("\n각 매개변수조합에서 수행한 교차검증의 평균들의 최댓값")
print("MAX : " + str(np.max(gs2.cv_results_['mean_test_score'])))
dt2 = gs2.best_estimator_
dt2.fit(train_input, train_target)
print("\n\n최적의 모델로 훈련한 테스트 점수 : " + str(dt2.score(test_input, test_target)))
