# 1. csv : http://bit.ly/wine_csv_data
# 2. pandas data -> numpy matrix
# 3. preprocessing
# 4. logistic regression -> score
# 5. DecisionTreeClassify
# 6. DT without preprocessing
# 7. plot all DTs

import pandas as pd
wine = pd.read_csv('http://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine[['class']].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state = 42
)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print("LR SCORE : ")
print("TRAIN SCORE : " + str(lr.score(train_scaled, train_target)))
print("TEST SCORE : " + str(lr.score(test_scaled, test_target)))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print("\nDT SCORE : ")
print("TRAIN SCORE : " + str(dt.score(train_scaled, train_target)))
print("TEST SCORE : " + str(dt.score(test_scaled, test_target)))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("DecisionTree")
plot_tree(dt)
plt.show()

# pruning (가지치기)
dt2 = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt2.fit(train_scaled, train_target)
print("\nDT (pruned) SCORE : ")
print("TRAIN SCORE : " + str(dt2.score(train_scaled, train_target)))
print("TEST SCORE : " + str(dt2.score(test_scaled, test_target)))

plt.figure(figsize=(10,7))
plt.title("DecisionTree (pruned)")  
plot_tree(dt2, feature_names = ['alcohol', 'sugar', 'pH'], filled = True,)
plt.show()

# DT는 preprocessing을 하지 않아도 됨
dt3 = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt3.fit(train_input, train_target)
print("\nDT (NO PROCESSING) SCORE : ")
print("TRAIN SCORE : " + str(dt3.score(train_input, train_target)))
print("TEST SCORE : " + str(dt3.score(test_input, test_target)))

plt.figure(figsize=(20,15))
plt.title("DecisionTree (NO PROCESSING)")
plot_tree(dt3, feature_names = ['alcohol', 'sugar', 'pH'], filled = True,)
plt.show()



