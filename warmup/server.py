# server.py

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
import jinja2
import os
import hashlib
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import torch
import logging
logging.basicConfig(level=logging.INFO)

if not os.path.exists("cache"):
    os.makedirs("cache")

app = Flask(__name__)

templateLoader = jinja2.FileSystemLoader(searchpath="./templates/")
templateEnv = jinja2.Environment(loader=templateLoader)

index_template = templateEnv.get_template("index.j2")
results_template = templateEnv.get_template("results.j2")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def crawl_page(url):
    if url is None:
        return None
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache = f"cache/page_text_{url_hash}.txt"
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            body = f.read()
        return body
    try:
        logging.info(f"Crawling {url}")
        response = requests.get(url,timeout=5)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.get_text()

        with open(cache, 'w') as f:
            f.write(body)

        return body
    except Exception as e:
        logging.error(f"Error crawling {url}: {e}")
        return None

def get_stories(n=500):
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)
    top_stories = response.json()[:n]
    
    logging.info(f"Getting stories.")
    stories = []

    def get_story(story_id):
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        story_response = requests.get(story_url)
        story_data = story_response.json()
        story_title = story_data.get('title')

        story_body = crawl_page(story_data.get('url'))
        if story_body is None:
            story_body = ""

        text = story_body
        # remove lines with < 200 characters
        text = ' '.join([line for line in text.split('\n') if len(line) > 200])
        text = story_title + '\n' + text
        vec = get_embeddings(text)

        return {
            'id': story_data.get('id'),
            'title': story_title,
            'body': text,
            'url': story_data.get('url'),
            'vec': vec,
        }
    with ThreadPoolExecutor(10) as executor:
        stories = list(executor.map(get_story, top_stories))
    
    # add original ranks
    for i, story in enumerate(stories):
        story['original_rank'] = i + 1
    
    return stories


# re-rank
def re_rank(stories, profile):
    logging.info(f"Ranking stories.")
    profile_vec = get_embeddings(profile)
    for story in stories:
        story['score'] = util.pytorch_cos_sim(profile_vec, story['vec']).item()
    stories.sort(key=lambda x: x['score'], reverse=True)
    for i, story in enumerate(stories):
        story['rank'] = i + 1
        story['change'] = story['original_rank'] - story['rank']
    return stories

# get embeddings
def get_embeddings(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    # save to cache
    cache_file = f"cache/{text_hash}.npy"
    if os.path.exists(cache_file):
        #convert to tensor in gpu
        l = np.load(cache_file)
        l = torch.tensor(l)
        if torch.cuda.is_available():
            l = l.cuda()
        return l
    
    logging.info(f"Getting embeddings for {text}")
    sentences = [text]
    embeddings = model.encode(sentences,convert_to_tensor=True)

    # save to cache
    np.save(cache_file, embeddings.cpu().numpy())

    return embeddings


@app.route('/')
def index():
    return index_template.render()

@app.route('/news', methods=['POST'])
def get_news():
    input = request.form
    if 'profile' not in input:
        return jsonify({'error': 'Invalid input'}), 400
    num = 10
    if 'number' in input:
        num = int(input['number'])
    if num > 500:
        num = 500
    if num < 1:
        num = 1
    
    stories = get_stories( num )
    
    stories = re_rank(stories, input['profile'])

    # take only title, url, original_rank and rank
    stories = [{'title': story['title'], 'url': story['url'], 'original_rank': story['original_rank'], 'rank': story['rank'], 'change': story['change'], 'score': story['score']} for story in stories]
    
    return results_template.render(results=stories)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)