from re import findall

# Objective: label article plain text as retrieved from RSS scrape

class RSSScrape:
    
    def __init__(self, input):
        self.input = input
        self.sources = self.__sources()
        self.matched = 0
        self.unmatched = 0
        # bias tallies
        self.extreme_left = 0
        self.left = 0
        self.left_center = 0
        self.least = 0
        self.right_center = 0
        self.right = 0
        self.extreme_right = 0
        # factualness tallies
        self.very_low = 0
        self.low = 0
        self.mixed = 0
        self.mostly_factual = 0
        self.high = 0
        self.very_high = 0
        
    def __sources(self):
        with open('../../database/sources.csv', 'r') as sources:
            return sources.read()
            
    def __get_text_from(self, link):
        # TODO
        pass
        
    def __tally(self, header):
        bias = header.split(',')[1]
        if bias == 'extreme_left':
            self.extreme_left += 1
        elif bias == 'left':
            self.left += 1
        elif bias == 'left_center':
            self.left_center += 1
        elif bias == 'least_biased':
            self.least += 1
        elif bias == 'right_center':
            self.right_center += 1
        elif bias == 'right':
            self.right += 1
        elif bias == 'extreme_right':
            self.extreme_right += 1
        factualness = header.split(',')[2]
        if factualness == 'very_low':
            self.very_low += 1
        elif factualness == 'low':
            self.low += 1
        elif factualness == 'mixed':
            self.mixed += 1
        elif factualness == 'mostly_factual':
            self.mostly_factual += 1
        elif factualness == 'high':
            self.high += 1
        elif factualness == 'very_high':
            self.very_high += 1
            
    def __summarize(self):
        s = '\nKnown sources: %d\nUnknown sources %d\n\n' % (self.matched, self.unmatched)
        s += 'EL: %d\nL: %d\nLC: %d\nleast: %d\nRC: %d\nR %d\nER: %d\n\n' % (
            self.extreme_left, self.left, self.left_center, self.least, self.right_center,
            self.right, self.extreme_right)
        s += 'EV: %d\nL: %d\nM: %d\nMF: %d\nH: %d\nVH %d\n' % (
            self.very_low, self.low, self.mixed, self.mostly_factual, self.high, self.very_high)
        print(s)
        
    def send_training_articles_to(self, directory):
        # TODO output labeled articles
        with open(self.input, 'r') as input:
            lines = [l.strip() for l in input.readlines()]
            article_titles = lines[0::4]
            for i, title in enumerate(article_titles):
                article_link = lines[4 * i + 1]
                article_text = self.__get_text_from(article_link)
                article_source = str(lines[4 * i + 2]).split('/')[2].replace('www.', '')
                source_matches = findall('%s[^\n]+\n' % article_source, self.sources)
                if not source_matches: 
                    self.unmatched += 1
                    continue
                self.matched += 1
                header_split = source_matches[0].split(',')
                del header_split[1:3]
                article_header = ','.join(header_split)
                self.__tally(article_header)
                print(i, article_header.strip())
        self.__summarize()
            
s = RSSScrape('../../scrapes/gnews_scrape.1.txt')
s.send_training_articles_to(None)
