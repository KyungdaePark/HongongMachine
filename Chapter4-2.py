#SGD Classifier
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish[['Species']].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state = 42
)
train_target = train_target.ravel()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# SGD Classifier
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log', max_iter = 10, random_state = 42)
sc.fit(train_scaled, train_target)
print("FISH_INPUT -> SCALED -> SGD(log, max_iter = 10)")
print("TRAIN SCORE : " + str(sc.score(train_scaled, train_target)))
print("TEST SCORE : " + str(sc.score(test_scaled, test_target)))

# 점진적 학습
sc.partial_fit(train_scaled, train_target)
print("\nFISH_INPUT -> SCALED -> SGD(log, max_iter = 10) -> Partial_Fit")
print("TRAIN SCORE : " + str(sc.score(train_scaled, train_target)))
print("TEST SCORE : " + str(sc.score(test_scaled, test_target)))

# 점진적 학습을 얼마나 해야 과소적합/과대적합이 일어나지 않을까?

import numpy as np
new_sc = SGDClassifier(loss = 'log', random_state = 42)
train_score = []
test_score = []
classes = np.unique(train_target) # fit 없이 partial_fit만을 위해 unique()함수 用

for _ in range(0,300):
    new_sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(new_sc.score(train_scaled, train_target))
    test_score.append(new_sc.score(test_scaled, test_target))
    
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('TRAIN / TEST SCORE')
plt.show()

# epoch = 100 이 좋겠군.
sc2 = SGDClassifier(loss = 'log', max_iter = 100, tol = None, random_state = 42) # tol이 None이어야 max_iter만큼 반복함.
sc2.fit(train_scaled, train_target)
print("\nFISH_INPUT -> SCALED -> SGD(log, max_iter = 100)")
print("TRAIN SCORE : " + str(sc2.score(train_scaled, train_target)))
print("TEST SCORE : " + str(sc2.score(test_scaled, test_target)))

sc3 = SGDClassifier(loss = 'hinge', max_iter = 100, tol = None, random_state = 42) # tol이 None이어야 max_iter만큼 반복함.
sc3.fit(train_scaled, train_target)
print("\nFISH_INPUT -> SCALED -> SGD(hinge), max_iter = 100)")
print("TRAIN SCORE : " + str(sc3.score(train_scaled, train_target)))
print("TEST SCORE : " + str(sc3.score(test_scaled, test_target)))



