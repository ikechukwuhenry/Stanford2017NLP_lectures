# from torchtext.data.utils import get_tokenizer
#  disable SSL 
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# All the codes above are not essential but to allow nltk work
# on apple silicon

# Preprocessing: Tokenization, stopword removal, stemming, Rare word removal

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
tokens = ['I', 'am', 'reading', 'a', 'book', 'now', '.', 'I', 'Love',
          'to', 'read', 'books', '!']

filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
print(filtered_tokens)

# Stemming
# Stemming reduce words to their base form
# import these modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print(stemmed_tokens)

# Remove rare words
from nltk.probability import FreqDist

freq_dist = FreqDist(stemmed_tokens)
threshold = 1
common_tokens = [token for token in stemmed_tokens if freq_dist[token] > threshold]
print(common_tokens)