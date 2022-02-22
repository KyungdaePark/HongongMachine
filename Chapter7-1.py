# 이제  tensorflow를 이용해 딥러닝을 할것임
# 기본적인 딥러닝 모델은 확률적 경사 하강법 + 로지스틱 회귀 모델임

# TF에서 데이터 받아오기 : (유명한 MNIST데이터 - 60000개의 28*28 이미지로 이루어져 있음)
from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()
# 또한 타깃 데이터는 각 레이블 별로 6000개씩 10개임.

# 이미지 10개 정도 그려보자.
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

# 우선 이 샘플들을 SGD Classifier를 이용해 훈련해보자.
# (훈련 샘플을 3차원에서 2차원으로, 0~255를 0~100으로 정규화 해야함)
# max_iter를 늘려가며 점수 확인 (cross_validate 이용)

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1,28*28)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
import numpy as np

sg = SGDClassifier(loss='log', max_iter=5, random_state = 42)
scores = cross_validate(sg, train_scaled, train_target, n_jobs=-1)
print("확률적 경사 하강법의 훈련 점수 (MAX_ITER = 5) :")
print(np.mean(scores['test_score']))

sg = SGDClassifier(loss='log', max_iter=9, random_state = 42)
scores = cross_validate(sg, train_scaled, train_target, n_jobs=-1)
print("\n확률적 경사 하강법의 훈련 점수 (MAX_ITER = 9) :")
print(np.mean(scores['test_score']))

sg = SGDClassifier(loss='log', max_iter=20, random_state = 42)
scores = cross_validate(sg, train_scaled, train_target, n_jobs=-1)
print("\n확률적 경사 하강법의 훈련 점수 (MAX_ITER = 20) :")
print(np.mean(scores['test_score']))

# MAX_ITER를 늘려도 점수가 크지 않다.
# Logistic Regression : z = w1 * f1 + w2 * f2 + ... + b

# SGD + LR을 기반으로 두는 인공 신경망은 큰 성능을 발휘하지 못 할것 같지만
# 인공 신경망 모델에는 좋은 기능들이 있음

# 고수준 API인 keras를 사용해보자.
# 딥러닝에서는 교차검증을 하지 않고 직접 검증 세트를 train_test_split() 메서드를 이용해
# 따로 덜어서 사용함 (CV로 하면 샘플이 너무 많고 오래 걸리기 때문)

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state=42
)

# A. 밀집층 생성
dense = keras.layers.Dense(10, activation='softmax', input_shape = (784,))
# 10 : 뉴런의 개수 ( = label 수 )
# softmax : 다중 분류이므로 결과값을 도출하기 전 통과하는 함수는 softmax (이진분류라면 sigmoid)
# == 활성화 함수 (activation function)
# input_shape : 입력으로 들어올 크기 (28*28 = 784)

# B. 이제 신경망 모델을 만들어보자. Sequential 클래스 사용
model = keras.Sequential(dense)

# C. Compile() 메서드로 손실 함수와 측정 지표 설정
model.compile(loss = "sparse_categorical_crossentropy", metrics = "accuracy")
# 원-핫 인코딩 ( 해당 클래스의 타깃값만 1, 나머지는 다 0으로 만드는 작업)을 하지 않고 훈련하기 위해
# sparse 사용, categorical : 다중분류 / binary : 이진분류
# accuracy : 매 epoch마다 정확도를 확인하고 싶어서 추가, (loss는 기본)
# (One Hot Encoding 은 훈련하는 모델이 사진/문자인 경우 숫자로 바꾸기 위해 사용)

# D. fit
model.fit(train_scaled, train_target, epochs = 5)

# E. Evaluate
model.evaluate(val_scaled, val_target)





