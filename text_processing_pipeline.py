# Import libraries
from torch.utils.data import Dataset, DataLoader

# Create a class
class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return self.text[idx]
    

from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Initialize and tokenize the text
tokenizer = get_tokenizer("basic_english")
# tokens = tokenizer(text)


def preprocess_sentences(sentences):
    preprocessed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = tokenizer(sentence)
        return tokens

def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    encoded_sentences = X.toarray()
    return encode_sentences, vectorizer


def extract_sentences(data):
    sentences = re.findall(r'[A-Z][^.!?]*[.!]', data)
    return sentences
    

def text_processing_pipeline(text):
    tokens = preprocess_sentences(text)
    encoded_sentences, vectorizer = encode_sentences(tokens)
    dataset = TextDataset(encode_sentences)
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer


# dataset = TextDataset(["",""])
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
text_data = 'This is the first text data. And here is another one.'
sentences = extract_sentences(text_data)
dataloaders, vectorizer = [text_processing_pipeline(text) for text in sentences]
print(next(iter(dataloaders))[0, :10])