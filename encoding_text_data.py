# curled from campus.datacamp.com
# Deep Learning for Text with PyTorch

import torch

def one_hot_encoding(vocab):
    print("inside function")
    vocab_size = len(vocab)
    one_hot_vectors = torch.eye(vocab_size)
    one_hot_labels = {word: one_hot_vectors[i] for i, word in enumerate(vocab)}
    return one_hot_vectors, one_hot_labels


if __name__ == "__main__":
    # example without using a function
    vocab = ['Cat', 'dog', 'rabbit']
    vocab_size = len(vocab)
    one_hot_vectors = torch.eye(vocab_size)
    one_hot_dict = {word: one_hot_vectors[i] for i, word in enumerate(vocab)}
    print(one_hot_dict)

    # using a function
    text = "I LIKE THE NEW MOVE!"
    text_list = text.split()
    one_hot_text, one_hot_labels = one_hot_encoding(text_list)
    print(one_hot_labels)

