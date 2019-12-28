from gensim.models import KeyedVectors
from re import split, findall
from numpy import ndarray, array
from random import randint

class Word:

    # Initialize with a string representation of the word
    def __init__(self, string, w2vm):
        self.string = string
        self.vector = self.__get_vector(w2vm)

    # Get a word2vec vector for this word, provided the loaded model
    def __get_vector(self, w2vm):
        
        # Return the vector if the word is in the vocabulary
        if self.string in w2vm:
            return w2vm.word_vec(self.string)
        # Return None if the word is not in the vocabulary
        else:
            return None
        
        # This will generate a random 1x300 matrix, useful in testing
        # return ndarray((1, 300), buffer=array([float(randint(0.0, 5.0)) for i in range(0, 300)]))

class Article:

    # Regular expression used to find words in sentences
    WORD_REGEX = "[^a-zA-Z]*([a-zA-Z']+)[^a-zA-Z]*"

    # Initialize with path to text file
    def __init__(self, path, w2vm):
        self.path = path
        self.words = self.__get_words(w2vm)
        self.vector = self.__get_vector(w2vm)
    
    # Function to get a list of words
    def __get_words(self, w2vm):
        print('Collecting words from article and generating vectors...')
        words = []
        with open(self.path, 'r') as i:
            for string in findall(self.WORD_REGEX, i.read()):
                words.append(Word(string.lower(), w2vm))
        return words

    # Function to get a word2vec representation for the article
    def __get_vector(self, w2vm):
        print('Creating vector representation for entire article...')
        total = ndarray((1, 300), buffer=array([0 for i in range(0, 300)]))
        for word in self.words:
            vector = word.vector
            if vector is not None:
                total += vector
        return total / len(self.words)

print('Loading word2vec model (this may take a minute or two)...')
w2vm = KeyedVectors.load_word2vec_format('classifiers/google_news', binary=True)
article = Article('articles/eg_text.txt', w2vm)
print(article.vector)

