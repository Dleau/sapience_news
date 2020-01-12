from bs4 import BeautifulSoup
from urllib.request import urlopen
from time import sleep
from os.path import exists
from datetime import datetime

# Objective: scrape an RSS news feed and write article titles, links, and sources to log

class Scraper:

    TAP = 'https://news.google.com/news/rss'
    
    def __init__(self, path):
        self.log = Scraper.Log(path)
        self.links = set()
        
    def activate(self):
        if self.log.already_exists():
            print('This log exists, importing existing links\n')
            for link in self.log.get_existing_links():
                self.links.add(link.strip())
        while True:
            additions = 0
            for item in self.__get_items():
                if item.link not in self.links:
                    self.log.write(item.title, item.link, item.source) # comment
                    self.links.add(item.link)
                    additions += 1
            print(datetime.now().time())
            print('Adding %d new articles to %s' % (additions, self.log.path))
            print('%d articles are on file' % len(self.links))
            print('Sleeping until next iteration\n')
            sleep(60)
        
    def __get_xml(self):
        with urlopen(self.TAP) as tap:
            return tap.read()
            
    def __get_soup(self):
        return BeautifulSoup(self.__get_xml(), 'xml')
        
    def __get_items(self):
        return [Scraper.Item(i) for i in self.__get_soup().findAll('item')]
        
    class Item:
    
        def __init__(self, xml_item):
            self.xml_item = xml_item
            self.title = self.__get_title()
            self.link = self.__get_link()
            self.source = self.__get_source()
            
        def __get_title(self):
            return str(self.xml_item.title.text)
            
        def __get_link(self):
            return str(self.xml_item.link.text)
            
        def __get_source(self):
            return str(self.xml_item.source).split('"')[1].split('"')[0]
            
    class Log:
    
        def __init__(self, path):
            self.path = path
            
        def already_exists(self):
            return exists(self.path)
            
        def get_existing_links(self):
            with open(self.path, 'r') as existing_log:
                yield from existing_log.readlines()[1::4]
                
        def write(self, title, link, source):
            with open(self.path, 'a') as log:
                log.write("%s\n%s\n%s\n\n" % (title, link, source))
            
        
s = Scraper('/Users/dbordeleau/Desktop/sapience/scrapes/gnews_scrape.2.txt')
s.activate()