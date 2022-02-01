# Cross Validate
# 훈련 세트의 20% 만큼 검증 세트로 나누기
# 훈련 64% 검증 16% 테스트 20%

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
dt = DecisionTreeClassifier()
dt.fit(sub_input, sub_target)
print("\nDecisionTree")
print("SUB SCORE : " +str(dt.score(sub_input, sub_target)))
print("VAL SCORE : " +str(dt.score(val_input, val_target)))

# cv, k-fold cv
from sklearn.model_selection import cross_validate
cv = cross_validate(dt, train_input, train_target)
print("\nDecisionTree, CV")
print("SCORE : " +str(cv['test_score']))
# 기본 5-fold cv
# VAL이 계속 바뀌며 5번에 걸쳐 검증하므로 5개의 결과값이 나온다.




