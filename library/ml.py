from gensim.models import KeyedVectors
from re import split, findall
from numpy import ndarray, array
from random import randint
from os import listdir
from os.path import join
from pandas import DataFrame, set_option
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pickle import load, dump
from sys import exit

class Word:
    # Initialize with a string and trained word2vec model
    def __init__(self, string, w2vm):
        self.string = string
        self.vector = self.__get_vector(w2vm)

    # Get a word2vec vector for this word, provided the loaded model
    def __get_vector(self, w2vm):
        # Return the vector if the word is in the vocabulary
        if self.string in w2vm.model:
            return w2vm.model.word_vec(self.string)
        # Return None if the word is not in the vocabulary
        else:
            return None

        # This will generate a random 1x300 matrix, very useful in testing
        #return ndarray((1, 300), buffer=array([float(randint(0.0, 5.0)) for i in range(0, 300)]))
        #return ndarray((1, 300), buffer=array([(1.0) for i in range(0, 300)]))


class Article:
    # Regular expression used to find words in sentences
    WORD_REGEX = "[^a-zA-Z]*([a-zA-Z']+)[^a-zA-Z]*"

    def __init__(self, path):
        self.path = path
        self.words = None
        self.vector = None

    def get_vector(self, w2vm):
        print('Representing article %s as a vector' % self.path)
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
        print('Identifying words in unseen article %s' % self.path)
        words = []
        with open(self.path, 'r') as i:
            for string in findall(super().WORD_REGEX, i.read()):
                words.append(Word(string.lower(), w2vm))
        print('Identifying words in %s' % self.path)
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
        self.source, self.bias, self.factualness, self.country = self.__header.split(',')
        self.vector = self.__get_vector(w2vm)

    # Function to get a list of words
    def __get_words(self, w2vm):
        print('Identifying words in training article %s' % self.path)
        words = []
        with open(self.path, 'r') as i:
            body_text = ''.join(i.readlines()[3:])
            for string in findall(super().WORD_REGEX, body_text):
                words.append(Word(string.lower(), w2vm))
        print('%d words were identified' % len(words))
        return words

    # Function to get training header
    def __get_header(self):
        with open(self.path, 'r') as i:
            return i.readlines()[1].strip()

    # Function to get a word2vec representation for the article
    def __get_vector(self, w2vm):
        return super().get_vector(w2vm)

class W2VClassifier:

    # Initialize with an input path
    def __init__(self, in_path):
        print('Loading pre-trained word2vec model (this may take a couple of minutes)...')
        self.model = KeyedVectors.load_word2vec_format(in_path, binary=True)

class Classifier:

    # Initialize with input path or output path
    def __init__(self, in_path=None, out_path=None):
        self.in_path = in_path
        self.out_path = out_path
        self.model = None
        self.confidence = None

    # Load a model from the input path
    def load(self):
        if not self.in_path:
            print('Cancelling classifier load; provide an input path during initialization')
            exit(0)
        print('Loading random forest classifier from %s' % self.in_path)
        self.model, self.confidence = load(open(self.in_path, 'rb'))

    # Create a random forest classifier from a directory of article vectorization data
    def create(self, directory, w2vm):
        if not self.out_path:
            print('Cancelling classifier creation; provide an output path during initialization')
            exit(0)
        print('Creating random forest classifier from training articles in %s' % directory)
        dict = self.create_dictionary(directory, w2vm)
        columns = list(dict.keys())
        frame = DataFrame(dict, columns=columns)
        # set_option('display.max_rows', None) # to print entire dataframe
        print('Data frame:')
        print(frame)
        x, y = frame[columns[:-1]], frame[columns[-1]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)
        self.model = model
        self.confidence = metrics.accuracy_score(y_test, model.predict(x_test))
        print('Saving classifier to %s' % self.out_path)
        dump((self.model, self.confidence), open(self.out_path, 'wb'))

    # Generate a list of TrainingArticles from a directory of files
    def get_training_articles_from(self, directory, w2vm):
        for path in [join(directory, name) for name in listdir(directory)]:
            yield TrainingArticle(path, w2vm)

    # Empty method for creation of dictionary
    def create_dictionary(self, directory, w2vm):
        raise Exception

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
            print('Mapping %s vectorization to dictionary' % article.path)

            if len(article.words) < 10: continue # TODO new line! get rid of NaN?

            vector = article.vector.tolist()[0] + [self.SCALE[article.bias]]
            for i, element in enumerate(vector[:-1]):
                dict[str(i)].append(round(element, 2))
                # TODO look into rounding
                '''
                round to 3 -> 0.63, n=298
                round to 2 -> 0.64, n=298
                '''
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
            print('Mapping %s vectorization to dictionary' % article.path)

            if len(article.words) < 10: continue # TODO new line! get rid of NaN?

            vector = article.vector.tolist()[0] + [self.SCALE[article.factualness]]
            for i, element in enumerate(vector[:-1]):
                dict[str(i)].append(round(element, 1))
                # TODO look into rounding
                '''
                round to 3 -> 0.74, n=298
                round to 2 -> 0.75, n=298

                round to 2 -> 0.77, n=~2000
                round to 1 -> 0.76, n=~2000
                '''
            dict['factualness'].append(vector[-1])
        return dict

def main():

    # Use this line for random vector generation (very fast, useful in testing)
    # w2vm = None
    # Use this line for word2vec vector generation (slow, takes a few minutes)
    w2vm = W2VClassifier(in_path='../../classifiers/google_news')

    # e.g., Create a factualness classifier
    # c = FactualnessClassifier(out_path='./test.model')
    c = FactualnessClassifier(out_path='./factualness.classifier')
    # c = BiasClassifier(out_path='./factualness.classifier')
    c.create(directory='../../ta/', w2vm=w2vm)

    # e.g., Load a factualness classifier
    # c = FactualnessClassifier(in_path='./test.model')
    # c.load()

    print('Classifier confidence:', c.confidence)

# main()
