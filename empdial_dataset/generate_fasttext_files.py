# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import numpy as np

# ------------------------- IMPLEMENTATION -----------------------------------

if __name__ == "__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))
    for data_type in ["train", "test"]:
        contexts = np.load(f"{dirname}/{data_type}/contexts.npy", allow_pickle=True)
        emotions = np.load(f"{dirname}/{data_type}/emotions.npy", allow_pickle=True)
        with open(f"{dirname}/{data_type}/fasttext_classifier.txt", "w") as f:
            for emotion, context in zip(emotions, contexts):
                line = f"__label__{emotion} " + " </ds> ".join(context) + "\n"
                f.write(line)

#-----------------------------------------------------------------------------