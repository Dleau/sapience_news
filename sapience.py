from gensim.models import KeyedVectors
from re import split, findall
from numpy import ndarray, array
from random import randint
from os import listdir
from os.path import join

class Word:
    # Initialize with a string and trained word2vec model
    def __init__(self, string, w2vm):
        self.string = string
        self.vector = self.__get_vector(w2vm)

    # Get a word2vec vector for this word, provided the loaded model
    def __get_vector(self, w2vm):
        '''
        # Return the vector if the word is in the vocabulary
        if self.string in w2vm.model:
            return w2vm.model.word_vec(self.string)
        # Return None if the word is not in the vocabulary
        else:
            return None
        '''
        # This will generate a random 1x300 matrix, very useful in testing
        return ndarray((1, 300), buffer=array([float(randint(0.0, 5.0)) for i in range(0, 300)]))
        
class Article:
    # Regular expression used to find words in sentences
    WORD_REGEX = "[^a-zA-Z]*([a-zA-Z']+)[^a-zA-Z]*"
    
    def __init__(self, path):
        self.path = path
        self.words = None
        self.vector = None
        
    def get_vector(self, w2vm):
        # print('Creating vector representation for entire article...')
        total = ndarray((1, 300), buffer=array([0 for i in range(0, 300)]))
        for word in self.words:
            vector = word.vector
            if vector is not None:
                total += vector
        return total / len(self.words)

class UnseenArticle(Article):
    # Initialize with path to text file and trained word2vec model
    def __init__(self, path, w2vm):
        super().__init__(path)
        self.words = self.__get_words(w2vm)
        self.vector = self.__get_vector(w2vm)
    
    # Function to get a list of words
    def __get_words(self, w2vm):
        # print('Collecting words from article and generating vectors...')
        words = []
        with open(self.path, 'r') as i:
            for string in findall(super().WORD_REGEX, i.read()):
                words.append(Word(string.lower(), w2vm))
        return words

    # Function to get a word2vec representation for the article
    def __get_vector(self, w2vm):
        return super().get_vector(w2vm)
        
class TrainingArticle(Article):
    # Initialize with path to text file and trained word2vec model
    def __init__(self, path, w2vm):
        super().__init__(path)
        self.words = self.__get_words(w2vm)
        self.__header = self.__get_header()
        self.source, self.bias, self.factualness, self.country = self.__header.split(', ')
        self.vector = self.__get_vector(w2vm)
        
    # Function to get a list of words
    def __get_words(self, w2vm):
        # print('Collecting words from %s and generating vectors...' % self.path)
        words = []
        with open(self.path, 'r') as i:
            body_text = ''.join(i.readlines()[2:])
            for string in findall(super().WORD_REGEX, body_text):
                words.append(Word(string.lower(), w2vm))
        return words
        
    # Function to get training header
    def __get_header(self):
        with open(self.path, 'r') as i:
            return i.readlines()[0].strip()
            
    # Function to get a word2vec representation for the article
    def __get_vector(self, w2vm):
        return super().get_vector(w2vm)

class Classifier:

    # Initialize with input path or output path
    def __init__(self, in_path=None, out_path=None):
        self.in_path = in_path
        self.out_path = out_path
        self.model = None
        
    # Load a model from the input path
    def load(self):
        raise Exception
      
    # Create a model from a directory of articles, save to output path  
    def create_from(self, directory, w2vm):
        raise Exception
        
    # Generate a list of TrainingArticles from a directory of files
    def get_training_articles_from(self, directory, w2vm):
        for path in [join(directory, name) for name in listdir(directory)]:
            yield TrainingArticle(path, w2vm)
        
class W2VClassifier(Classifier):

    # Initialize with an input path
    def __init__(self, in_path=None):
        super().__init__(in_path=in_path)
        
    # Load a word2vec model from the input path
    def load(self):
        print('Loading pre-trained word2vec model (this may take a couple of minutes)...')
        self.model = KeyedVectors.load_word2vec_format(self.in_path, binary=True)
        
class BiasClassifier(Classifier):

    # Convert bias string to integer for use in matrix
    SCALE = {
        'extreme_left': 0,
        'left': 1,
        'left_center': 2,
        'least_biased': 3,
        'right_center': 4,
        'right': 5,
        'extreme_right': 6
    }

    # Initialize with an input path or output path
    def __init__(self, in_path=None, out_path=None):
        super().__init__(in_path=in_path, out_path=out_path)
        
    # Load an existing model from the input path, set self.model
    def load(self): # TODO
        pass
        
    # Create a new model from the directory of articles and word2vec model,
    # save the new model to the output path
    def create_from(self, directory, w2vm): # TODO
        articles = super().get_training_articles_from(directory, w2vm)
        for article in articles:
            # use article.vector and article.bias to create vector
            print(self.SCALE[article.bias])
    
class FactualnessClassifier(Classifier):

    # Convert factualness string to integer for use in matrix
    SCALE = {
        'very_low': 0,
        'low': 1,
        'mixed': 2,
        'mostly_factual': 3,
        'high': 4,
        'very_high': 5
    }

    # Initialize with an input path or output path
    def __init__(self, in_path=None, out_path=None):
        super().__init__(in_path=in_path, out_path=out_path)
        
    # Load an existing model from the input path, set self.model
    def load(self): # TODO
        pass
        
    # Create a new model from the directory of articles and word2vec model,
    # save the new model to the output path
    def create_from(self, directory, w2vm): # TODO
        articles = super().get_training_articles_from(directory, w2vm)
        for article in articles:
            # use article.vector and article.factualness to create vector
            print(self.SCALE[article.factualness])

def main():    
    c = FactualnessClassifier(out_path='test.model')
    c.create_from('articles', 'model')
    
main()
