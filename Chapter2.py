# Chapter1에서는 test와 target이 같아서 score의 의미가없었음
# 시험문제의 정답을 알려주고 시험문제가 바뀌지 않고 그대로 나온 격
# 그래서 target과 data를 섞은 후 일부분만 떼서 사용할 것임. ()

from sklearn.neighbors import KNeighborsClassifier
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l,w] for l,w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

#여기서부터는 잘못된 경우를 보여줌

train_input = fish_data[:35]
train_target = fish_target[:35]
wrong_test_input = fish_data[35:]
wrong_test_target = fish_target[35:]

kn = KNeighborsClassifier()
kn = kn.fit(train_input, train_target)
wrong_score = kn.score(wrong_test_input, wrong_test_target)
print(wrong_score)

# 잘못된 이유는, fish_data의 0~34 원소는 bream을 35~48 원소는 smelt를 가지고 있는데
# 학습은 bream만을 이용했으면서 테스트는 smelt만을 이용해 결과를 도출하려고 하기 때문에 score가 0이 나옴
# 이런 경우를 샘플링 편향이라고 하며, 이를 해결하기 위해서는 bream과 smelt를 적절히 섞어 학습해야 함.

#data 와 target은 같은 index를 이용해 1:1 매칭되어 있으므로 하나의 배열로 묶어서 관리할 것임
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# random하게 섞을 것임. seed는 42로 정의
np.random.seed(42)
index = np.arange(49) #0부터 48까지의 원소가 들어간 배열 index를 만듬
np.random.shuffle(index)

# 랜덤하게 섞인 index 배열들의 원소를 이용해 input/target_arr의 원소를 무작위로 접근함
train_input = input_arr[index[:35]]
train_targt = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
print(train_input)
print(input_arr[13], train_input[0])

# 잘 섞였는지 확인을 위해 2차원 그래프로 그려볼 것
import matplotlib.pyplot as plt
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 섞은 데이터들을 학습시키고 잘 맞추는지 확인
kn = kn.fit(train_input, train_target)
score = kn.score(test_input, test_target)
print(score)

print(kn.predict(test_input))
print(test_target)





