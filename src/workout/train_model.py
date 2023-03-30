import tensorflow as tf
# 내장 함수 사용

# # 학습 준비
# model.compile(
#     optimizer = 'adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # 학습 실행
# model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=3
# )

samples = [
    '너 오늘 이뻐 보인다',
    '나는 오늘 기분이 더러워',
    '끝내주는데, 좋은 일이 있나봐',
    '나 좋은 일이 생겼어',
    '아 오늘 진짜 짜증나',
    '환상적인데,정말 좋은거 같아'
]
labels = [
    [1],[0],[1],[1],[0],[1]
]
tokenizer = tf.keras.preprocessing.text.Tokenizer()
print("# tokenizer ---> ", tokenizer)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
print("# sequences ---> ", sequences)

word_index = tokenizer.word_index
print("# word_index ---> ", word_index)

batch_size = 2
num_epochs = 100
vocab_size = len(word_index) + 1
emb_size = 128 # 임베딩 층
hidden_dimension = 256 # 은닉층
output_dimension = 1 # 출력층

# Sequential API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size))
model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))
model.add(tf.keras.layers.Dense(units=hidden_dimension, activation='relu'))
model.add(tf.keras.layers.Dense(units=output_dimension, activation='sigmoid'))
'''
입력값을 임베딩하는 임베딩 층을 모델에 추가하였음.
이후에 Lambda 층을 추가하여 임베딩 층의 출력값을 평균을 내어 하나의 벡터로 만들었다.
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''
이진 분류 문제이므로 손실 함수로 binary_crossentropy를 사용하였다.
옵티마이저는 adam 최적화 알고리즘을 사용하였다.
모델 성능 평가 지표로는 정확도(accuracy)로 지정하였다.
'''

model.fit(sequences, labels, epochs=num_epochs, batch_size=batch_size)

# Functional API
inputs = tf.keras.Input(shape=(4,))
embed_output = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size)(inputs)
pooled_output = tf.reduce_mean(embed_output, axis=1)
hidden_layer = tf.keras.layers.Dense(units=hidden_dimension, activation='relu')(pooled_output)
outputs = tf.keras.layers.Dense(units=output_dimension, activation='sigmoid')(hidden_layer)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)
model.fit(sequences, labels, epochs=num_epochs, batch_size=batch_size)

# Subclassing API
class CustomModel(tf.keras.Model):
    def __init__(self, vocab_size, emb_size, hidden_dimension, output_dimension):
        super(CustomModel, self).__init__(name='my_model')
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden_dimension, activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=output_dimension, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense_hidden(x)
        return self.dense_output(x)
    
model = CustomModel(vocab_size=vocab_size, emb_size=emb_size, hidden_dimension=hidden_dimension, output_dimension=output_dimension)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)
model.fit(sequences, labels, epochs=num_epochs, batch_size=batch_size)
