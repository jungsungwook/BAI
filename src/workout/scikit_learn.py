import sklearn
from sklearn.datasets import load_iris

iris_dataset = load_iris()
keys = iris_dataset.keys()
for key in keys:
    print("# ",key,"---> ",iris_dataset[key])
    
'''
data : 4개의 특징값을 가지고 있는, 실제 데이터
데이터의 형태는 (150,4)이다.
따라서 150개의 데이터가 있고, 각 데이터는 4개의 특징값을 가지고 있다.

feature_names : 데이터의 특징값이 의미하는 바를 나타내는 문자열 리스트
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
결과를 보면, 각 특징값이 의미하는 바는 다음과 같다.
sepal length : 꽃받침의 길이
sepal width : 꽃받침의 너비
petal length : 꽃잎의 길이
petal width : 꽃잎의 너비

target : 데이터의 레이블값을 나타내는 리스트
target_names : 레이블값이 의미하는 바를 나타내는 문자열 리스트

DESCR : 데이터셋에 대한 설명을 나타내는 문자열
해당 데이터에 대한 전체적인 요약 정보를 보여준다.
'''
'''
사이킷런을 이용하면 학습 데이터를 대상으로 학습 데이터와 평가 데이터로 쉽게 분리할 수 있다.
'''

from sklearn.model_selection import train_test_split
train_input, test_input, train_label, test_label = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=42)
'''
위 코드는 사이킷런의 train_test_split() 함수를 이용하여 학습 데이터와 평가 데이터를 분리하였다.
분리 비율은 정의한 test_size 매개변수의 값에 따라 결정된다.
0.25는 전체 학습 데이터의 25%를 평가 데이터로 사용하겠다는 의미이다.
random_state 매개변수는 난수 발생 시드를 의미한다.
'''

'''
* 지도 학습
지도 학습이란, 데이터에 대한 정답이 주어진 상태에서 학습을 진행하는 방식이다.
'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # 여기서 n_neighbors는 예측하고자 하는 데이터와 가까운 k개의 데이터를 찾는다는 의미이다.
knn.fit(train_input, train_label) # 학습 데이터를 이용하여 학습을 진행한다.
KNeighborsClassifier(
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    metric_params=None,
    n_jobs=1,
    n_neighbors=1,
    p=2,
    weights='uniform'
)

'''
위에서 학습시킨 knn 모델을 이용하여 새로운 데이터에 대한 예측을 진행해보자.
'''
import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])

result = knn.predict(new_input)
print(result) # [1]이 출력된다. 즉, 1번 레이블을 가진 데이터로 예측하였다.

'''
이번엔 맨 위에서 만든 평가 데이터를 이용하여 모델의 성능을 평가해보자.
'''
predict_label = knn.predict(test_input)
print(predict_label) # 이 결과값만으로는 정확도가 어느정도인지 확인 할 수 없음
print(np.mean(predict_label == test_label)) # 1.0이 출력된다. 즉, 100%의 정확도를 가진다는 의미이다.

'''
* 비지도 학습
비지도 학습이란, 데이터에 대한 정답이 주어지지 않은 상태에서 학습을 진행하는 방식이다.
'''
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3) # 여기서 n_clusters는 군집의 개수를 의미한다.

k_means.fit(train_input) # 학습 데이터를 이용하여 학습을 진행한다.

KMeans(
    algorithm='auto',
    copy_x=True,
    init='k-means++',
    max_iter=300,
    n_clusters=3,
    n_init=10,
    random_state=None,
    tol=0.0001,
    verbose=0
)

'''
비지도 학습은 라벨이 필요 없기 때문에, 입력 데이터만으로 학습을 진행한다.
군집이 라벨이라고 생각하면 된다.

위에서 학습한 k_means 모델을 이용하여 새로운 데이터에 대한 예측을 진행해보자.
'''
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
prediction = k_means.predict(new_input)
print(prediction) # [1]이 출력된다. 즉, 1번 군집에 속한다는 의미이다.

predict_cluster = k_means.predict(test_input)
print(predict_cluster)

# 군집의 라벨을 바꿔서 정확도를 확인해보자.
# 군집의 라벨을 바꾸는 이유는 라벨이 0, 1, 2로 되어있는데, 이는 무작위로 지정된 것이다. 따라서, 정확도를 확인하기 위해서는 라벨을 바꿔야 한다.
np_arr = np.array(predict_cluster)
np_arr[np_arr == 0], np_arr[np_arr == 1], np_arr[np_arr == 2] = 3, 4, 5
np_arr[np_arr == 3] = 1
np_arr[np_arr == 4] = 0
np_arr[np_arr == 5] = 2
predict_label = np_arr.tolist()
print(predict_label) # [1, 0, 2, 2, 0, 1, 2, 0, 0, 1, 1, 2, 1, 0, 2, 2, 0, 1, 0, 0, 1, 2, 0, 2, 2, 1, 1, 2, 0, 0]

print(np.mean(predict_label == test_label)) # 0.9473684210526315 출력된다. 즉, 94.74%의 정확도를 가진다는 의미이다.