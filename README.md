# sapience news

ML project to predict bias and factualness in news articles.

**Interpreter operations**

- importlib.reload(packagename)

```python
>>> from w2v import MODEL # will import word2vec model
...
>>> from importlib import reload # enables reload of modules
...
>>> reload(atn); atn.main(MODEL, '../database/psample.csv') # reload and run
```
