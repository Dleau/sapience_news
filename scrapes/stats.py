from sys import argv

# scrape as argv[1]

titles = set()
articles = set()
sources = set()

with open(argv[1], 'r') as scrape:
    sequence = 0
    for i, line in enumerate(scrape.readlines()):
        if sequence == 0: # title line
            # print('title:' + line.strip())
            titles.add(line.strip())
        elif sequence == 1: # article link line
            # print('article:' + line.strip())
            articles.add(line.strip())
        elif sequence == 2: # source link line
            # print('source:' + line.strip())
            sources.add(line.strip())
        sequence = (sequence + 1) % 4

temp = set()
for source in sources:
    domain = '.'.join(source.split('.')[1:])
    if domain not in temp: temp.add(domain)
print(len(temp))
