import spacy

nlp = spacy.load('en_core_web_sm')
sentence = "Natural Language Processing is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."
doc = nlp(sentence)

word_tokens = [token.text for token in doc]
sentence_tokens = [sent.text for sent in doc.sents]
print(word_tokens)
print(sentence_tokens)