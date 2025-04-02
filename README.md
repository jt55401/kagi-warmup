# Setup

```
cd warmup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```
* browser to http://localhost:5000

# Thought process

1) Hash out the basics - identified basic way to do simliarity using hf sentence transformers + cosine similiarty (from past experience) This is not the most sophisticated, but, it's easy, and we're timeboxed.
2) Get data from hackernews, pretty simple, learned api and used basic python
3) Did a few passes of smaller datasets, mocked up results in cli
4) Wrapped it up into an api server using flask and templates using jinja
5) A few test/fix/performance cycles:
    * added caching at various stages so it would not be so slow to test.
    * found some pages to have garbage coming back, added simple filtering rules
    * added timeouts to make sure we don't hang on slow requests
6) pylint - fixed most of it
7) Added documentation
8) Still had an hour left, so, experimented with a few different models, sentence-transformers/all-MiniLM-L6-v2 seemed to work well for this.

# Future thoughts

1) Break articles into sentences, and match on those instead?
2) All sorts of things to do around better crawling/parsing/text extraction
3) Probably better models to find as well?
4) Remove logging
5) Make app 12-factor
6) Add user input cleaning