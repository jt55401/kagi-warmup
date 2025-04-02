# server.py

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import jinja2
import os
import hashlib
import numpy as np
import requests
import json
from bs4 import BeautifulSoup

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
    cache = f"cache/page_text_${url_hash}.txt"
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            body = f.read()
        return body

    response = requests.get(url)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    body = soup.get_text()

    with open(cache, 'w') as f:
        f.write(body)

    return body

def get_stories(n=500):
    cache = "cache/topstories.json"
    if os.path.exists(cache):
        with open(cache, 'r') as f:
            top_stories = json.load(f)
        if len(top_stories) >= n:
            return top_stories[:n]

    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)
    top_stories = response.json()[:n]
    
    stories = []
    for story_id in top_stories:
        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        story_response = requests.get(story_url)
        story_data = story_response.json()

        story_body = crawl_page(story_data.get('url'))
        if story_body is None:
            story_body = ""

        stories.append({
            'id': story_data.get('id'),
            'title': story_data.get('title'),
            'body': story_body,
            'url': story_data.get('url'),
            'original_rank': len(stories) + 1,
        })
    
    # save to cache
    with open(cache, 'w') as f:
        json.dump(stories, f)
    
    return stories


# re-rank
def re_rank(stories, profile):
    profile_vec = get_embeddings(profile)
    for story in stories:
        story['score'] = (story['vec'] @ profile_vec.T).item()
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
        return np.load(cache_file)
    
    sentences = [text]
    embeddings = model.encode(sentences)

    # save to cache
    np.save(cache_file, embeddings)

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
    
    # get embeddings for each story, in parallel
    def get_story_embeddings(story):
        text = story['title'] + ' ' + story['body']
        story['vec'] = get_embeddings(text)
        return story
    with ThreadPoolExecutor() as executor:
        stories = list(executor.map(get_story_embeddings, stories))

    stories = re_rank(stories, input['profile'])

    # take only title, url, original_rank and rank
    stories = [{'title': story['title'], 'url': story['url'], 'original_rank': story['original_rank'], 'rank': story['rank'], 'change': story['change'], 'score': story['score']} for story in stories]
    
    return results_template.render(results=stories)

if __name__ == '__main__':
    app.run(debug=True)