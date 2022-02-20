
import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
# fruits는 (300,100,100)으로 되어있음.
# 0~99는 사과, 100~199는 파인애플, 200~299는 바나나임

# 사과하나 출력?
plt.imshow(fruits[0],cmap='gray_r')
plt.show()

# 우선 각 샘플들의 픽셀값의 평균을 구해보자(axis=1,2)
# 혹은 각 샘플들을 (100 * (10000) 으로 바꾸고 axis=1로 해도됨)
apple = fruits[:100].reshape(-1,100*100)
pineapple = fruits[100:200].reshape(-1,100*100)
banana = fruits[200:300].reshape(-1,100*100)


# 각 평균들을 히스토그램으로 그려보자 ( 빈도 )
alpha = 0.8
plt.hist(np.mean(apple,axis=1), alpha = alpha)
plt.hist(np.mean(pineapple,axis=1), alpha = alpha)
plt.hist(np.mean(banana,axis=1), alpha = alpha)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# 결과를 확인하니 바나나는 구분하기 쉬운데 사과와 파인애플이 어렵네
# 그럼 각 샘플의 평균이 아니라 과일마다 각 픽셀값들의 평균을 구해보자
# axis = 0
fig, axs = plt.subplots(1,3,figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana,axis=0))
plt.show()

fruit_mean = []

fruit_mean.append(np.mean(apple,axis=0).reshape(100,100))
fruit_mean.append(np.mean(pineapple,axis=0).reshape(100,100))
fruit_mean.append(np.mean(banana, axis =0).reshape(100,100))

fig, axs = plt.subplots(1,3,figsize=(10,7))
axs[0].imshow(fruit_mean[0], cmap='gray_r')
axs[1].imshow(fruit_mean[1], cmap='gray_r')
axs[2].imshow(fruit_mean[2], cmap='gray_r')
plt.show()

abs_diff = []
abs_mean = []
for idx in range(3):
    abs_diff.append(np.abs(fruits - fruit_mean[idx])) 
    abs_mean.append(np.mean(abs_diff[idx], axis=(1,2))) # [모든 샘플들] 과 [각 과일의 평균]의 차이의 평균
    
# abs_diff는 (300,100,100), abs_mean은 (300,)
# 이제 [각 샘플]과 [각 과일들이 전체 샘플들과 얼마나 차이나는지에 대한 평균값(300,)]의 차이가 가장 적은
# 그러니까 [각 과일 평균]과 차이가 가장 적은
# 예를들어 사과 평균 모습이랑 가장 닮은 샘플을 추려내자

apple_index = np.argsort(abs_mean[0])[:100]
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
  for j in range(10):
    axs[i,j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')
    axs[i,j].axis('off')
plt.show()

# 파인애플 출력
pineapple_index = np.argsort(abs_mean[1])[:100]
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
  for j in range(10):
    axs[i,j].imshow(fruits[pineapple_index[i*10+j]], cmap='gray_r')
    axs[i,j].axis('off')
plt.show()

# 바나나 출력
banana_index = np.argsort(abs_mean[2])[:100]
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
  for j in range(10):
    axs[i,j].imshow(fruits[banana_index[i*10+j]], cmap='gray_r')
    axs[i,j].axis('off')
plt.show()
    

