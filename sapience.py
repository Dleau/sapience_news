from gensim.models import KeyedVectors
from re import split, findall
from numpy import ndarray, array
from random import randint
from os import listdir
from os.path import join
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pickle import load, dump

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
        self.accuracy = None

    # Load a model from the input path
    def load(self):
        self.model, self.accuracy = load(open(self.in_path, 'rb'))

    # Create a random forest classifier from a directory of article vectorization data
    def create(self, directory, w2vm):
        dict = self.create_dictionary(directory, w2vm)
        columns = list(dict.keys())
        frame = DataFrame(dict, columns=columns)
        print(frame)
        x, y = frame[columns[:-1]], frame[columns[-1]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)
        self.model = model
        self.accuracy = metrics.accuracy_score(y_test, model.predict(x_test))
        dump((self.model, self.accuracy), open(self.out_path, 'wb'))

    # Generate a list of TrainingArticles from a directory of files
    def get_training_articles_from(self, directory, w2vm):
        for path in [join(directory, name) for name in listdir(directory)]:
            yield TrainingArticle(path, w2vm)

    # Empty method for creation of dictionary
    def create_dictionary(self, directory, w2vm):
        raise Exception

class W2VClassifier(Classifier):

    # Initialize with an input path
    def __init__(self, in_path=None):
        super().__init__(in_path=in_path)
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

    # Create a dictionary of article vectorization data
    def create_dictionary(self, directory, w2vm):
        dict = {**{str(i): [] for i in range(0, 300)}, **{'bias': []}}
        articles = super().get_training_articles_from(directory, w2vm)
        for article in articles:
            vector = article.vector.tolist()[0] + [self.SCALE[article.bias]]
            for i, element in enumerate(vector[:-1]):
                dict[str(i)].append(round(element, 3)) # TODO look into rounding
            dict['bias'].append(vector[-1])
        return dict

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

    # Create a dictionary of article vectorization data
    def create_dictionary(self, directory, w2vm):
        dict = {**{str(i): [] for i in range(0, 300)}, **{'factualness': []}}
        articles = super().get_training_articles_from(directory, w2vm)
        for article in articles:
            vector = article.vector.tolist()[0] + [self.SCALE[article.factualness]]
            for i, element in enumerate(vector[:-1]):
                dict[str(i)].append(round(element, 3)) # TODO look into rounding
            dict['factualness'].append(vector[-1])
        return dict

def main():
    w2vm = None #W2VClassifier(in_path='classifiers/google_news')
    c = FactualnessClassifier(out_path='./test.model')
    c.create(directory='articles/', w2vm=w2vm)
    print(c.model)

main()
