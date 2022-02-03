# 앙상블 트리의 랜덤 포레스트, 엑스트라 트리, 그레디언트 부스팅, 히스토그램 기반 그레디언트 부스팅
# 랜덤 포레스트는 결정트리를 사용하는데 훈련 세트를 부트스트램 샘플로 사용함 (중복), 최적의 트리를 찾음
# 엑스트라 트리는 모든 훈련 세트를 다 사용함. 대신 무작위로 트리를 찾아봄, 최적의 트리를 찾으려고 하지 않기때문에 속도가 빠름
# 또 많은 샘플을 해보기때문에 과대적합이 적고 점수가 높게나옴
# 그레디언트 부스팅은 경사 하강법 + 결정트리, 과대적합이 거의 일어나지 않지만 천천히 움직이기 때문에 속도가 느림
# 히스토그램 기반 그레디언트 부스팅은 훈련세트를 256조각으로 나누어 최적의 샘플을 찾기 때문에 그레디언트 부스팅의 속도 문제를 해결함

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
wine = pd.read_csv('http://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine[['class']].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state = 42
)
train_target = train_target.ravel()

# RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs = -1)
print("\nRandomForest Train/Test Score : ")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# RF 의 특성 중요도
rf.fit(train_input, train_target)
print("\nRandom Forest Feature_Importances : ")
print(rf.feature_importances_)

# RF의 자체 모델 평가 : 훈련 테스트를 모두 사용하지 않고, 사용하지 않은 부분으로 검증함
rf2 = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 42)
rf2.fit(train_input, train_target)
print("\nRandomForest Out of Bag Score : ")
print(rf2.oob_score_)


# Extra Tree
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(et, train_input, train_target, return_train_score = True, n_jobs = -1)
print("\nExtraTree Train/Test Score : ")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# ET 의 특성 중요도
et.fit(train_input, train_target)
print("\nExtraTree Feature_Importances : ")
print(et.feature_importances_)

# GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state = 42)
scores = cross_validate(gb, train_input, train_target, return_train_score = True, n_jobs = -1)
print("\nGradient Boosting Train/Test Score : ")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# GB의 학습률을 증가시키고 트리의 개수를 늘려보자. 과대적합이 잘 일어나지 않음
gb2 = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.2, random_state = 42,) # 기본값 100, 0.1
scores = cross_validate(gb2, train_input, train_target, return_train_score = True, n_jobs = -1)
print("\nGradient Boosting -> 학습률 & 트리의 개수 ↑ Train/Test Score : ")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 매개변수 중 subsamples의 값을 1보다 작게하면 미니배치/ 확률적 경사 하강법이 됨. 지금은 배치 경사 하강법임.

# GB의 특성 중요도
gb2.fit(train_input, train_target)
print("\nGradient Boosting Feature_Importances : ")
print(gb2.feature_importances_)

# Histogram based GradientBoosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state = 42)
scores = cross_validate(hgb, train_input, train_target, return_train_score = True, n_jobs = -1)
print("\nHistogram based Gradient Boosting Train/Test Score : ")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# HGB는 GB의 n_estimators (트리의 개수)를 건드리지 않고 반복 횟수(부스팅 ?, max_iter)를 건드림

# HGB의 특성중요도는 다르게 구함. 랜덤하게 어떤 특성이 들어갔을때 가장 성능이 좋은지를 판단해 그때의 특성 중요도값을 출력함
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result_hgb_train = permutation_importance(hgb, train_input, train_target, n_repeats = 10, random_state = 42, n_jobs = -1)
# n_repeats 는 랜덤하게 섞을 횟수, 기본값 5
# result에는 importances_mean, importances_std, importances가 들어있음
print("\nHistogram based Gradient Boosting Feature_Importances : ")
print(result_hgb_train.importances_mean)

result_hgb_test = permutation_importance(hgb, test_input, test_target, n_repeats = 10, random_state = 42, n_jobs = -1)
print("\nHistogram based Gradient Boosting -> 랜덤하게 섞은 횟수 ↑ Train/Test Score : ")
print(result_hgb_test.importances_mean)

print("\nHGB TEST SCORE :")
print(hgb.score(test_input, test_target))


