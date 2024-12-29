from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer()
corpus = ['this is the first document.', 'This document is the second document.',
          'And this is the third one.', 'Is this the first document?']

X1 = vectorizer.fit_transform(corpus)
print(X1.toarray())
print(vectorizer.get_feature_names_out())

# Term frequency vectorizer
# Lessfrequent words have higher value and more frequent words have lower value
# Scores the importance of words in a document
# Rare words have a higher score
# Common ones have a lower score
tfVectorizer = TfidfVectorizer()
corpus = ['This is the first document.', 'This document is the second document.', 
          'And this is the third one.', 'Is this the first document?']
X2 = tfVectorizer.fit_transform(corpus)
print(X2.toarray())
print(tfVectorizer.get_feature_names_out)