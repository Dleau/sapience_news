from bs4 import BeautifulSoup
from urllib.request import urlopen
from time import sleep

class GoogleNewsFeed:

    URL = 'https://news.google.com/news/rss'

    def __init__(self):
        self.news = self.__news()
        
    def __xml(self):
        with urlopen(self.URL) as source:
            return source.read()
            
    def __soup(self):
        return BeautifulSoup(self.__xml(), "xml")
        
    def __news(self):
        return self.__soup().findAll("item")
        
class NewsItem:

    def __init__(self, element):
        self.__element = element
        self.title = self.__title()
        self.link = self.__link()
        self.source_url = self.__source_url()
        
    def __title(self):
        return str(self.__element.title.text)
        
    def __link(self):
        return str(self.__element.link.text)
        
    def __source_url(self):
        return str(self.__element.source).split('"')[1].split('"')[0]
    
def get_items():
    for item in [NewsItem(i) for i in GoogleNewsFeed().news]:
        yield (item.title, item.link, item.source_url)
        
def log(title, link, source):
    with open('log.txt', 'a') as log:
        log.write("%s\n%s\n%s\n\n" % (title, link, source))
        
def main():
    links = set()
    new_items = 0
    total_items = 0
    while True:
        print('Getting items from Google News feed')
        for title, link, source in get_items():
            if link in links: continue
            log(title, link, source)
            links.add(link)
            new_items += 1
            total_items += 1
        print('Collected %d new items' % new_items)
        print('Total items: %d\n' % total_items)
        new_items = 0
        sleep(60)
        
main()
        