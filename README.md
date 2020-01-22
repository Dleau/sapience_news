# sapience news

ML project to predict bias and factualness in news articles.

**Interpreter operations**


```python
>>> from w2v import MODEL # will import word2vec model
...
>>> import atn # import atn.py
...
>>> from importlib import reload # enables reload of modules
...
>>> reload(atn); atn.main(MODEL, '../database/psample.csv') # reload and run
```
