from sklearn.feature_extraction.text import CountVectorizer
import re

# Create a list of stopwords
stop_words = set(stopwords.words("english"))

# Initialize the tokenizer and stemmer
tokenizer = get_tokenizer("basic_english")
stemmer = PorterStemmer() 

# Complete the function to preprocess sentences
def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
		# Tokenize the sentence
        tokens = tokenizer(sentence)
		# Remove stop words
        tokens = [token for token in tokens if token not in stop_words]
		# Stem the tokens
        tokens = [stemmer.stem(token) for token in tokens]
        processed_sentences.append(' '.join(tokens))
    return processed_sentences

processed_shakespeare = preprocess_sentences(shakespeare)
print(processed_shakespeare[:5]) 


def preprocess_sentences(sentences):
    preprocessed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = tokenizer(sentence)

def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    encoded_sentences = X.toarray()
    return encode_sentences, vectorizer


def extract_sentences(data):
    sentences = re.findall(r'[A-Z][^.!?]*[.!]', data)
    return sentences