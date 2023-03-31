from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
'''
토크나이징 부분은 잘 아는 부분이라 예제 코드만 작성
'''
# 단어 단위 토크나이저
sentence = "Natural Language Processing is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."
print(word_tokenize(sentence))

# 문장 단위 토크나이저
paragraph = "Natural Language Processing is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural-language generation."
print(sent_tokenize(paragraph))