from sys import argv, maxsize
from csv import reader, field_size_limit
from numpy import ndarray, array
from re import findall
from pandas import DataFrame, set_option
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from pickle import load, dump
from gensim.models import KeyedVectors
from math import log
 
from ml import Word, W2VClassifier

class AllTheNewsCSV:
    
    PUBLICATIONS = {
        # source: (n, bias, factualness)
        # bias: extreme left, left, left center, least biased, right center, right, extreme right
        # factualness: very low, low, mixed, mostly factual, high, very high
        'Atlantic': (0, 2, 4), # left center, high
        'Breitbart': (1, 6, 2), # extreme right, mixed
        'Business Insider': (2, 2, 4), # left center, high
        'Buzzfeed News': (3, 2, 2), # left center, mixed
        'CNN': (4, 1, 3), # left, mostly factual
        'Fox News': (5, 5, 2), # right, mixed
        'Guardian': (6, 2, 4), # left center, high
        'National Review': (7, 5, 3), # right, mostly factual
        'New York Post': (8, 4, 2), # right center, mixed
        'New York Times': (9, 2, 4), # left center, high
        'NPR': (10, 2, 5), # left center, very high
        'Reuters': (11, 3, 5), # least biased, very high
        'Talking Points Memo': (12, 1, 3), # left, mostly factual
        'Vox': (13, 1, 3), # left, mostly factual
        'Washington Post': (14, 2, 4) # left center, high
    }
    
    def __init__(self, path):
        field_size_limit(maxsize) # Increase maximum field size, CSV is very large
        self.path = path
        self.n = 0 # number of articles
        self.hashed_words = self.__hashed_words()
        
        
    def __hashed_words(self): # {'word': 6, 'word2': 4, etc.}
        # first pass of articles, tally words
        # the number of articles in which this word appears!
        hash = {}
        print('Tallying words of ATN articles from %s' % self.path)
        with open(self.path, 'r') as master_text:
            master_csv = reader(master_text, delimiter=',')
            next(master_csv) # skip the header line
            for i, e in enumerate(master_csv):
                self.n += 1 # number of articles
                # ,id,title,publication,author,date,year,month,url,content
                _, _, _, _, _, _, _, _, _, content = e
                ta = TrainingArticle(None, None, None, content, None)
                seen = set() # set of seen words in this article
                for string in [word.string for word in ta.words]:
                    if string in seen: # word was already seen in this article, skip
                        continue
                    else: # add the word to the seen set
                        seen.add(string)
                    if string in hash:
                        temp = hash[string]
                        hash[string] = temp + 1
                    else:
                        hash[string] = 1
        return hash
        
    def articles(self, w2vm):
        print('Labeling ATN articles from %s' % self.path)
        with open(self.path, 'r') as master_text:
            master_csv = reader(master_text, delimiter=',')
            next(master_csv) # skip the header line
            for i, e in enumerate(master_csv):
                # ,id,title,publication,author,date,year,month,url,content
                _, _, title, pub, author, _, _, _, _, content = e
                n, bias, factualness = self.PUBLICATIONS[pub]
                yield TrainingArticle(n, bias, factualness, content, w2vm)
                
    def idf(self, string): # DONE!
        # get the Inverse Document Frequency of a word string
        # IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
        if string in self.hashed_words:
            return log(self.n, self.hashed_words[string])
        else:
            return 0
                
class TrainingArticle:

    WORD_REGEX = "[^a-zA-Z]*([a-zA-Z']+)[^a-zA-Z]*"

    def __init__(self, n, bias, factualness, content, w2vm):
        self.n = n
        self.bias = bias
        self.factualness = factualness
        self.content = content
        self.w2vm = w2vm
        self.words = self.__words()
        
    def tf(self, string):
        # get the term frequency of a word string
        # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
        pass
        
    # experimental method!      
    # TODO I think this needs idf data          
    def vector(self):
        total = ndarray((1, 300), buffer=array([0 for i in range(0, 300)]))
        for word in self.words:
            vector = word.vector
            if vector is not None: 
                total += vector
        return total / len(self.words)
        
    def __words(self):
        words = []
        for s in findall(self.WORD_REGEX, self.content):
            words.append(Word(s.lower(), self.w2vm))
        return words
        
class BiasDataFrame:

    # round to 2, n = 142,232, confidence @ 0.56
    # no rounting, n = 142,232, confidence @ 0.56

    def __init__(self, tas):
        self.__d = self.__create_from(tas)
        self.df = DataFrame(self.__d, columns=list(self.__d.keys()))
        
    def __create_from(self, tas): # list of training articles
        d = {**{str(i): [] for i in range(0, 300)}, **{'bias': []}}
        for i, ta in enumerate(tas):
            if len(ta.words) < 10: continue # skip articles less than n words
            print('Examining training article %d' % (i + 1), end='\r')
            vector = ta.vector().tolist()[0] + [ta.bias]
            for i, element in enumerate(vector[:-1]):
                # d[str(i)].append(round(element, 2)) # TODO play with rounding
                d[str(i)].append(element)
            d['bias'].append(vector[-1])
        return d
        
class BiasClassifier:

    def __init__(self, to, bdf): # BiasDataFrame
        self.to, self.bdf = to, bdf
        self.model, self.confidence = self.__create()
        
    def __create(self):
        # set_option('display.max_rows', None) # to print entire dataframe
        print(self.bdf.df)
        columns = list(self.bdf.df.columns)
        x, y = self.bdf.df[columns[:-1]], self.bdf.df[columns[-1]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        # model = RandomForestClassifier(n_estimators=100)
        model = RandomForestClassifier(n_estimators=1000) # TODO play w/
        model.fit(x_train, y_train)
        confidence = metrics.accuracy_score(y_test, model.predict(x_test))
        dump((model, confidence), open(self.to, 'wb'))
        return model, confidence
        
def main(w2vm, csv_path):

    # argv[1] as gnews model
    # argv[2] as master.csv
    # w2vm = W2VClassifier(argv[1])
    # tas = AllTheNewsCSV(argv[2]).articles(W2VM)
    
    #tas = AllTheNewsCSV(csv_path).articles(w2vm)
    #bdf = BiasDataFrame(tas)
    #bc = BiasClassifier('./test_model', bdf)
    #print(bc.confidence)
    
    tas = AllTheNewsCSV(csv_path)
    r = tas.idf('trump')
    print(r)
    
    # http://www.tfidf.com/
    
    
main(None, '../database/psample.csv')