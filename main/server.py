# server.py

import os
import hashlib
import logging
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
import jinja2
import numpy as np
import requests
from bs4 import BeautifulSoup
import torch


logging.basicConfig(level=logging.INFO)

if not os.path.exists("cache"):
    os.makedirs("cache")

app = Flask(__name__)

templateLoader = jinja2.FileSystemLoader(searchpath="./templates/")
templateEnv = jinja2.Environment(loader=templateLoader)

index_template = templateEnv.get_template("index.j2")


def get_queries_from_question():
    """Use a pretrained model to generate queries from a question."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question = "What is the capital of France?"
    queries = ["capital of France", "France capital", "Paris"]
    question_embedding = model.encode(question, convert_to_tensor=True)
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, query_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=3)
    return [queries[idx] for idx in top_results[1]]


@app.route("/")
def index():
    """
    Renders and returns the index page using the index template.

    Returns:
        str: The rendered HTML content of the index page.
    """
    return index_template.render()


@app.route("/queries", methods=["POST"])
def get_queries():
    """
    Takes a question, turns it into a few queries, and returns them.
    Returns:
        Response: JSON response containing the generated queries.
    Input:
        - Expects a POST request with form data containing:
            - 'question' (required): A string representing the user's question
            - 'number' (optional): An integer specifying the number of stories to fetch
              (default is 10, minimum is 1, maximum is 500)
    Output:
        - A list of strings
    Error Handling:
        - Returns a JSON response with an error message and a 400 status code
          if the 'question' key is missing from the input.
    """
    form_data = request.form
    if "question" not in form_data:
        return jsonify({"error": "Invalid input"}), 400
    num = 5
    if "number" in form_data:
        num = int(form_data["number"])
    if num > 10:
        num = 10
    if num < 1:
        num = 1

    question = form_data["question"]
    
    queries = get_queries_from_question(question)
    
    return results_template.render(results=queries)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
