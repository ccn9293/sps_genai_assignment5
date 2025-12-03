import spacy
import subprocess
import sys

def load_spacy_model(model_name="en_core_web_lg"):
    try:
        nlp=spacy.load(model_name)
    except OSError:
        print("Error")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        nlp=spacy.load(model_name)
    return nlp

class WordEmbedding:
    def __init__(self):
        self.nlp = load_spacy_model()

    def calculate_embedding(self, word: str):
        token = self.nlp(word)
        if token.has_vector:
            return token.vector.tolist() # might return long list
        else:
            raise ValueError("Error")

#from sklearn.metrics.pairwise import cosine_similarity
#def calculate_similarity(word1, word2):
   # return cosine_similarity([nlp(word1).vector], [nlp(word2).vector])[0][0]


