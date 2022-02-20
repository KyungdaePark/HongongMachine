import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)

# 각 클러스터가 어떤 그림을 그리는지 그려보자
import matplotlib.pyplot as plt


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols * ratio, rows * ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


draw_fruits(fruits[km.labels_ == 0])
draw_fruits(fruits[km.labels_ == 1])
draw_fruits(fruits[km.labels_ == 2])

# 클러스터의 중심(각 과일들의 평균값)을 그려보자.
# 평균값 정보는 km.cluster_centers_에 있음

draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# transform을 이용해 클러스터와 샘플들의 거리를 구해보자.
# 거리가 가장 가까운 label이 곧 km이 predict하는 label과 같다.

print(km.transform(fruits_2d[100:101]))  # fruits_2d[100]으로 하면 (10000, ) 의 배열이 전달되므로 인덱싱
# label 0이 제일 가깝네.

print(km.predict(fruits_2d[100:101]))
# 일치.


###
# 근데 우리는 지금까지 n_clusters 를 3으로 지정해서 함.
# 실전에서는 클러스터의 개수조차 알 수 없기 때문에 최적의 K 를 찾아야함.
# elbow 방법 : 클러스터의 중심과 샘플들간의 거리 ^2 의 합을 이니셔라고 하는데,'
# 이 이니셔는 클러스터의 개수가 많아질수록 작아짐
# 이니셔가 작아야 정확하겠지?
# 근데 이 그래프는 딱 팔꿈치처럼 꺾이는 순간이 있음 그때가 최적의 K
# km.inertia_
###

inertia = []
for k in range(2,7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2,7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()

# k=3일때가 최적이구나