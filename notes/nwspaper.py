from newspaper import build

for article in build('https://dentonrc.com/').articles:
    print(article.url)
    article.download()
    article.parse()
    print(article.text)
    