# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize

# ------------------------- IMPLEMENTATION -----------------------------------

def generate_glove_vectors(vocab, glove_fpath):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    with open(glove_fpath)as f:
        glove_data = f.readlines()
        if len(glove_data) == 0:
            raise ValueError("No glove data found in specified file!")
        embed_dim = len(glove_data[0].split()) - 1
        if embed_dim not in [25, 50, 100, 200, 300]:
            raise ValueError("Unexpected vector dimensions found in glove file!")

    vocab_size = len(vocab)
    embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
    
    num_pretrained = 0
    print("Finding pretrained glove vectors for vocabulary...")
    for line in tqdm(glove_data):
        line = line.split()
        word = line[0]
        if word in vocab:
            try:
                embeddings[vocab[word]] = np.array([float(val) for val in line[1:]])
            except ValueError:
                print("Glove vectors must be numeric values that can be converted to float!")
            num_pretrained += 1
    print("Done!")
    print('Percentage of vocabulary words pre-trained with glove vectors: %d (%.2f%%)' 
          % (num_pretrained, num_pretrained * 100.0 / vocab_size))
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fpath", type=str, default=None)
    cli_args = parser.parse_args()

    dirname = os.path.dirname(os.path.abspath(__file__))

    word_pairs = {
        "it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": 
        "did not", "you'd": "you would", "you're": "you are", "you'll": "you will", 
        "i'm": "i am", "they're": "they are", "that's": "that is", "what's": "what is", 
        "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
        "i'd": "i would", "aren't": "are not", "isn't": "is not", 
        "wasn't": "was not", "weren't": "were not", "won't": "will not", 
        "there's": "there is", "there're": "there are"
    }

    special_tokens = ["[PAD]", "[UNK]", "[EOS]", "[SOS]"]
    vocab = {token: i for i, token in enumerate(special_tokens)}

    with open(f"{dirname}/data.txt") as f:
        data = f.readlines()
        print("Generating vocabulary...")
        for row in tqdm(data):
            row = row.lower()
            for k, v in word_pairs.items():
                row = row.replace(k, v)
            tokens = word_tokenize(row)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        print(f"Done! Generated vocab size: {len(vocab)}")

    vocab_fpath = f"{dirname}/vocab.json"
    with open(vocab_fpath, "w") as f:
        json.dump(vocab, f)
        print(f"Vocab file saved to {os.path.abspath(vocab_fpath)}")

    # Save glove vectors for vocab
    if cli_args.glove_fpath is not None:
        print("")
        vocab_glove_fpath = f"{dirname}/glove_embed_matrix.npy"
        np.save(
            vocab_glove_fpath, 
            generate_glove_vectors(vocab, cli_args.glove_fpath)
        )
        print(f"Glove vectors saved at {os.path.abspath(vocab_glove_fpath)}")

# -----------------------------------------------------------------------------