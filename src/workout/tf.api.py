import tensorflow as tf
'''
API
TensorFlow2.0부터는 API를 하나로 통일하였다.
1. 이거 모드
    - 연산을 구성하면서 바로바로 값을 확인할 수 있다.
2. 모델 구축
    - 모델을 구축하고 학습시키는 방식이다.
    모델을 구축하는 방법은 다음과 같이 분류할 수 있다.
    - Sequential API
    - Functional API
    - Functional/Sequential API
        + Custom Layer
    - Subclassing API (Custom Model)
'''
# Sequential API
# 완전 연결 계층( fully connected layer )
model = tf.keras.Sequential()
model.add(layer=tf.keras.layers.Dense(64, activation='relu'))
model.add(layer=tf.keras.layers.Dense(64, activation='relu'))
model.add(layer=tf.keras.layers.Dense(10, activation='softmax'))
'''
Sequential API는 모델을 구성하는 계층을 순차적으로 쌓아서 모델을 구성하는 방식이다.
위와 같이 Sequential 인스턴스를 생성하고, add() 메서드를 사용해 계층을 추가하면 된다.
다만, 이 방식은 모델을 구성하는 계층이 순차적으로 쌓여야 하는 경우에만 사용할 수 있다.
예를 들어, VQA(사진과 질문이 입력값으로 주어지고 사진을 참고해 질문에 답하는 문제)의 경우 사진과 질문을 병렬로 처리해야 하므로 Sequential API를 사용할 수 없다.
'''

# Functional API
# 완전 연결 계층( fully connected layer )
inputs = tf.keras.Input(shape=(32, ))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
'''
위에서 만든 모델을 Functional API를 사용해 구현해본 것이다.
'''

# Custom Layer
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        self.hidden_dimension = hidden_dimension
        self.hidden_dimension2 = hidden_dimension2
        self.output_dimension = output_dimension
        super(CustomLayer, self).__init__()

    def build(self, input_shape):
        self.dense_layer1 = tf.keras.layers.Dense(units=self.hidden_dimension, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(units=self.hidden_dimension2, activation='relu')
        self.dense_layer3 = tf.keras.layers.Dense(units=self.output_dimension, activation='softmax')

    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        return self.dense_layer3(x)

'''
위와 같이 Custom Layer를 만들어서 Functional API 또는 Sequential API를 사용해 모델을 구축할 수 있다.
'''
model = tf.keras.Sequential()
model.add(CustomLayer(64, 64, 10))

# Subclassing API
class MyModel(tf.keras.Model):
    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        super(MyModel, self).__init__(name='my_model')
        self.dense_layer1 = tf.keras.layers.Dense(units=hidden_dimension, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(units=hidden_dimension2, activation='relu')
        self.dense_layer3 = tf.keras.layers.Dense(units=output_dimension, activation='softmax')

    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        return self.dense_layer3(x)