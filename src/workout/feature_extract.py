from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
'''
특징 추출.
텍스트 데이터에서 단어나 문장들을 어떤 특징 값으로 바꿔주는 것을 의미한다.
기존에 문자로 구성된 데이터를 모델에 적용할 수 있도록 특징을 뽑아 어떤 값으로 바꿔서 수치화 한다.
'''

text_data = [
    "나는 배가 고프다",
    "내일 점심 뭐먹지",
    "내일 공부 해야겠다",
    "점심 먹고 공부 해야지",
]

# 1. CountVectorizer
# CountVectorizer는 단순히 텍스트에서 횟수를 기준으로 특징을 추출하는 방법이다.
# 여기서 어떤 횟수를 기준으로 특징을 추출할지는 CountVectorizer의 파라미터로 설정할 수 있다.
count_vectorizer = CountVectorizer()
count_vectorizer.fit(text_data)
print(count_vectorizer.vocabulary_)

sentence = [text_data[0]]
print(count_vectorizer.transform(sentence).toarray())

# 2. TfidfVectorizer
# TfidfVectorizer는 TF-IDF라는 값을 사용해 텍스트에서 특징을 추출하는 방법이다.
# TF란 특정 단어가 하나의 데이터 안에서 등장하는 횟수를 의미한다.
# DF는 문서 빈도값으로, 특정 단어가 여러 데이터에서 자주 등장하는지를 알려주는 지표이다.
# IDF는 이 값에 역수를 튀해서 구할 수 있으며, 특정 단어가 다른 데이터에 등장하지 않을 수록 값이 커진다.
# 따라서 TF-IDF는 이 값을 곱해서 사용하므로, 어떤 단어가 해당 문서에 자주 등장하지만 다른 문서에서는 많이 없는 단어일수록 높은 값을 가지게 된다.
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(text_data)
print(tfidf_vectorizer.vocabulary_)

sentence = [text_data[3]]
print(tfidf_vectorizer.transform(sentence).toarray())

# 3. HashingVectorizer
# HashingVectorizer는 CountVectorizer와 동일한 방법이지만 텍스트를 처리할 때 해시 함수를 사용하기 때문에 실행 시간을 크게 줄일 수 있다.