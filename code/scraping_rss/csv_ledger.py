from urllib.request import urlopen
from re import findall

# Objective: use Media Bias Fact Check URLs to derive and store source URLs

class CSVLedger:

    def __init__(self, path):
        self.path = path

    BIAS = {
        # 'extreme_left' # we may need some of these!
        'LEFT BIAS': 'left',
        'LEFT-CENTER BIAS': 'left_center',
        'LEAST BIASED': 'least_biased',
        'RIGHT-CENTER BIAS': 'right_center',
        'RIGHT BIAS': 'right',
        # 'extreme_right' # we may need some of these!
    }
    
    FACTUALNESS = {
        # 'very_low' # we may need some of these!
        # 'low' # we may need some of these!
        'MIXED': 'mixed',
        'MOSTLY FACTUAL': 'mostly_factual',
        'HIGH': 'high',
        'VERY HIGH': 'very_high',
    }
        
    def transcribe_to(self, path):
        with open(self.path, 'r') as log:
            for line in log.readlines()[1:]:
                url, name, bias, factualness, country = line.split(',')
                source = CSVLedger.MBFCURL(url).source
                if not source: continue
                translated_bias = self.BIAS[bias]
                translated_factualness = self.FACTUALNESS[factualness]
                s = ','.join([source, url, name, translated_bias, translated_factualness, country]).strip()
                print(s)
                with open(path, 'a') as destination:
                    destination.write(s + "\n")
                
    class MBFCURL: # Media Bias Fact Check URL
    
        def __init__(self, url):
            self.url = url
            self.contents = self.__contents()
            self.source = self.__source()
            
        def __contents(self):
            return str(urlopen(self.url).read())
            
        def __source(self):
            matches = findall('Source:[^<]*<a href="([^"]+)"', self.contents)
            if not matches: return None
            return str(matches[0]).split('/')[2].replace('www.', '')
                
l = CSVLedger('/Users/dbordeleau/Desktop/sapience/labels/bias_labels.csv')
l.transcribe_to('./new_ledger.csv')
# <p>Source: <a href="https://news.abs-cbn.com/"