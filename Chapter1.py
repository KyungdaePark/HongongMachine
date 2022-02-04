import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# matplotlib를 이용해 그래프 그리기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30,600,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

#bream과 smelt를 fish로 zip을 이용해 묶음
#학습을 위해서 data와 target을 만듬. data는 실제 값, target은 0과 1 (0은 우리가 원하는 값, 1은 그 반대)
fish_data = [[l,w] for l,w in zip(length, weight)]

print(fish_data)

fish_target = [1]*35 + [0]*14
print(fish_target)

# sklearn의 fit 함수를 이용해 학습한다
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

#fish_data를 이용해 fish_target을 맞추는 확률을 계산 (점수)
print(kn.score(fish_data, fish_target))


#neightbors=n은 한 그룹을 몇개로 하냐 : 여기서 35개의 bream 14개의 smelt가 있는데 neighbors가 49가 되면 모든 원소들이 하나의 특성으로 파악됨
for n in range(5,50):
  kn.n_neighbors = n
  score = kn.score(fish_data, fish_target)
  if score<1:
    print(n,score)
    break


