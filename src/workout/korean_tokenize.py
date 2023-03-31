from konlpy.tag import Okt

okt = Okt()

text = "한글 자연어 처리는 재밌다 이제부터 열심히 해야지ㅎㅎㅎㅎ"
print(okt.morphs(text)) # 형태소 추출
print(okt.morphs(text, stem=True)) # 형태소 추출 + 어간 추출

print(okt.nouns(text)) # 명사만 추출
print(okt.phrases(text)) # 어절만 추출

print(okt.pos(text)) # 품사 태깅
# [('한글', 'Noun'), ('자연어', 'Noun'), ('처리', 'Noun'), ('는', 'Josa'), ('재밌다', 'Adjective'), ('이제', 'Noun'), ('부터', 'Josa'), ('열심히', 'Adverb'), ('해야지', 'Verb'), ('ㅎㅎㅎㅎ', 'KoreanParticle')]
print(okt.pos(text, join=True)) #형태소와 품사 태깅을 합쳐서 추출
# ['한글/Noun', '자연어/Noun', '처리/Noun', '는/Josa', '재밌다/Adjective', '이제/Noun', '부터/Josa', '열심히/Adverb', '해야지/Verb', 'ㅎㅎㅎㅎ/KoreanParticle']