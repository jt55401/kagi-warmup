# Warmup

## Setup

```
cd warmup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```
* browser to http://localhost:5000

## Thought process

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

## Future thoughts

1) Break articles into sentences, and match on those instead?
2) All sorts of things to do around better crawling/parsing/text extraction
3) Probably better models to find as well?
4) Remove logging
5) Make app 12-factor
6) Add user input cleaning



# Main Task

## Setup

```
cd main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```
* browser to http://localhost:5001


## Thought Process

1) Research existing models and/or datasets to train a model
  * https://microsoft.github.io/msmarco/ORCAS.html seems like it might be a good dataset to train a custom model with
  * Used search engines and "deep research" (Gemini and OpenAI - and checked results for accuracy) to find the best models for the purpose, preferring statistical or "small" language models over "large" language models. (due to latency constraint of 100ms)
  * Found and read a few papers on the subject
    * https://arxiv.org/abs/2002.02631 (this is about the reverse process, which isn't directly useful, but, good for ideas/data)
    * https://dl.acm.org/doi/abs/10.1162/coli_a_00010
    * https://ieeexplore.ieee.org/document/10508170 (2023, more recent, but not very relevant to this task - until it needs to scale)
  * Models considered
    * Phi-3 Mini (very optimized for low latency, and many hardware options via ONNX)
    * Mixtral (may be good, but requirements are higher, come back to it later?)
    * Llama 3 - specifically Llama3.2-1B or 8B
    * Gemma3-4B - very new, and designed to run on single gpu/lower end hardware.
    * DistilGPT2 - distilled version of an old model - might still be ok for this task.
    * T5-Small or T5-Base (or even Tiny? with their efficient variants also worth considering) - text-text models, also work with ONNX
    * prhegde/t5-query-reformulation-RL - this is specifically designed for a task such as what we're doing, and based on T5-Base. Advantage being it ensures the queries which are returned are more diverse.
2) Create a benchmarking script/notebook, and evaluate the models for speed and quality of results
3) Make a decision on model
4) Create a little webapp to demonstrate it

## Future thoughts

1) Excluded for this exercise, but, I think it'd be interesting to take one of the smaller models on the list, and fine tune it using ORCAS or even our own Kagi data.
2) Review preferred model licenses to ensure compatibility with end usecase.
3) Evaluate use case for streaming vs. batch mode