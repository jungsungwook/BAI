import tensorflow as tf
'''
TensorFlow의 케라스 모듈인 tf.kreas.layers 모듈을 사용해보자.
1. tf.keras.layers.Dense
'''
inputs = tf.keras.layers.Input(shape=(20, 1))
dense = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)
hidden = dense(inputs)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)
print(output, hidden)

'''
위 코드는 10개의 노드를 가지는 은닉층이 있고, 최종 출력 값은 2개의 노드가 있는 신경망 구조를 표현한 것이다.

다음 모듈을 사용해보자.
2. tf.keras.layers.Dropout
'''
inputs = tf.keras.layers.Input(shape=(20, 1))
dropout = tf.keras.layers.Dropout(rate=0.5)(inputs)
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(dropout)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)

'''
Dropout은 rate 확률값에 따라 노드들의 값을 0으로 만들어 학습데이터에 과적합되는 상황을 방지하기 위해 사용한다.
따라서 학습 시에만 사용하고, 테스트 시에는 사용하지 않는다.
위 코드는 이전에 사용한 신경망 구조에서 처음 입력값에 드롭아웃을 적용한 것이다.

3. tf.keras.layers.Conv1D
4. tf.keras.layers.MaxPool1D
합성곱 신경망을 구현하기 위해 사용한다.
'''
inputs = tf.keras.layers.Input(shape=(20, 1))
conv1d = tf.keras.layers.Conv1D(filters=10, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(inputs)
maxpool1d = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(conv1d)
flatten = tf.keras.layers.Flatten()(maxpool1d)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(flatten)