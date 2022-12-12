import json
import string
import nltk
from tqdm import tqdm
from pattern.text.en import singularize

stopwords = set(nltk.corpus.stopwords.words('english'))

with open("concepts_nrc_vad.json") as f:
    concepts = json.load(f)

with open("../empdial_dataset/data.txt") as f:
    data = f.readlines()

count, total = 0, 0
for line in tqdm(data):
    words = line.split(" ")
    for word in words:
        word = singularize(word.lower()).translate(str.maketrans('', '', string.punctuation))
        if word in stopwords:
            continue
        total += 1
        if word in concepts:
            count += 1

print(f"Percentage of words with concepts: {count * 100 / total:.2f}%")
