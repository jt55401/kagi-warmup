"""
LLM Benchmarking Script for comparing different models.

This script benchmarks various language models using the same set of questions,
tracking both load time and inference time for each model.

--- prompt to sketch it out ---

Benchmark the following models:
Phi-3, Llama3.2-1B, Llama3.2-9B, Gemma3-4B, DistilGPT2, T5-Base, T5-Small, T5-Tiny, T5-efficient-Base, T5-efficient-Small, T5-efficient-Tiny, and prhegde/t5-query-reformulation-RL
Make sure each model is run in most efficient way possible
Use the same questions as input for all models
Run each model for 10 iterations
Please also track the load time of each model (not the download), and unload the model fully from GPU afterwards.
Use ONNX for the models that support it
Save the results as a nice HTML file
"""
import gc
import os
import time
from typing import Dict, List

import pandas as pd
import torch
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline, GPT2Tokenizer
)
import plotly.express as px


# Use the following questions as input:
questions = [
    "In what year was the winner of the 44th edition of the Miss World competition born?",
    "Who lived longer, Nikola Tesla or Milutin Milankovic?",
    "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?",
    "Create a table for top noise cancelling headphones that are not expensive",
    "what are some ways to do fast query reformulation",
]


def clear_gpu_memory():
    """Clear GPU memory to ensure clean runs between models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def benchmark_model(model_info, questions, iterations=10):
    """Benchmark a model with given questions for multiple iterations."""
    model_name = model_info["name"]
    model_id = model_info["id"]
    model_type = model_info["type"]
    use_onnx = model_info.get("onnx", False)
    
    print(f"\nBenchmarking {model_name}...")
    
    results = []
    
    for i in range(iterations):
        clear_gpu_memory()
        
        # Load model and measure loading time
        start_time = time.time()
        
        if model_type == "causal":
            if use_onnx:
                model = ORTModelForCausalLM.from_pretrained(model_id)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            else:
                if model_id == "distilgpt2":
                    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                )
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            
        elif model_type == "seq2seq":
            if use_onnx:
                model = ORTModelForSeq2SeqLM.from_pretrained(model_id, device_map=device)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                )
            pipe = pipeline(
                "text2text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        
        loading_time = time.time() - start_time
        
        # Process each question
        for q_idx, question in enumerate(questions):
            start_time = time.time()
            
            if model_type == "causal":
                output = pipe(question, max_new_tokens=100, do_sample=False)[0]["generated_text"]
                # For causal models, the output includes the input, so we need to extract only the generated text
                if output.startswith(question):
                    output = output[len(question):].strip()
            else:
                output = pipe(question, max_length=100)[0]["generated_text"]
            
            inference_time = time.time() - start_time
            
            results.append({
                "model": model_name,
                "iteration": i + 1,
                "question_idx": q_idx + 1,
                "question": question,
                "output": output,
                "inference_time": inference_time,
                "loading_time": loading_time if q_idx == 0 else None  # Only record loading time once per iteration
            })
        
        # Explicitly delete model and tokenizer
        del model, tokenizer, pipe
        clear_gpu_memory()
    
    return results

models = [
    {"name": "Phi-3-mini", "id": "microsoft/phi-3-mini-4k-instruct", "type": "causal", "onnx": True},
    {"name": "Llama3.2-1B", "id": "meta-llama/Meta-Llama-3.2-1B", "type": "causal", "token": True},
    {"name": "Llama3.2-3B", "id": "meta-llama/Meta-Llama-3.2-3B", "type": "causal", "token": True},
    {"name": "Llama3.2-1B-Instruct", "id": "meta-llama/Meta-Llama-3.2-1B-Instruct", "type": "causal", "token": True},
    {"name": "Llama3.2-3B-Instruct", "id": "meta-llama/Meta-Llama-3.2-3B-Instruct", "type": "causal", "token": True},
    {"name": "Gemma3-1B", "id": "google/gemma-3-1b-it", "type": "causal", "token": True},
    {"name": "Gemma3-4B", "id": "google/gemma-3-4b-it", "type": "causal", "token": True},
    {"name": "DistilGPT2", "id": "distilgpt2", "type": "causal", "onnx": False},
    {"name": "T5-Base", "id": "t5-base", "type": "seq2seq", "onnx": True},
    {"name": "T5-Small", "id": "t5-small", "type": "seq2seq", "onnx": True},
    {"name": "T5-Tiny", "id": "t5-tiny", "type": "seq2seq", "onnx": True},
    {"name": "T5-efficient-Base", "id": "google/t5-efficient-base", "type": "seq2seq"},
    {"name": "T5-efficient-Small", "id": "google/t5-efficient-small", "type": "seq2seq"},
    {"name": "T5-efficient-Tiny", "id": "google/t5-efficient-tiny", "type": "seq2seq"},
    {"name": "T5-query-reformulation-RL", "id": "prhegde/t5-query-reformulation-RL", "type": "seq2seq"}
]

# Define all models to benchmark


# Run benchmarks
all_results = []
for model_info in models:
    try:
        results = benchmark_model(model_info, questions, iterations=10)
        all_results.extend(results)
    except Exception as e:
        print(f"Error benchmarking {model_info['name']}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# write to disk
results_df.to_csv("llm_benchmark_results.csv", index=False)


# Calculate aggregate statistics
stats_df = results_df.groupby("model").agg({
    "inference_time": ["mean", "std", "min", "max"],
    "loading_time": ["mean", "std", "min", "max"]
}).reset_index()

# Flatten the multi-index columns
stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

# Create HTML report
def generate_html_report(results_df, stats_df):
    """Generate an HTML report with benchmark results and visualizations."""
    # Create plotly figures
    # 1. Inference time comparison
    fig1 = px.bar(
        stats_df, 
        x="model", 
        y="inference_time_mean", 
        error_y="inference_time_std",
        title="Average Inference Time by Model",
        labels={"model": "Model", "inference_time_mean": "Inference Time (s)"}
    )
    
    # 2. Loading time comparison
    fig2 = px.bar(
        stats_df, 
        x="model", 
        y="loading_time_mean", 
        error_y="loading_time_std",
        title="Model Loading Time",
        labels={"model": "Model", "loading_time_mean": "Loading Time (s)"}
    )
    
    # 3. Box plot of inference times
    fig3 = px.box(
        results_df, 
        x="model", 
        y="inference_time", 
        title="Inference Time Distribution",
        labels={"model": "Model", "inference_time": "Inference Time (s)"}
    )
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Benchmarking Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; }}
            .plot {{ margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">LLM Benchmarking Results</h1>
            
            <h2 class="mt-4">Aggregate Statistics</h2>
            <div class="table-responsive">
                {stats_df.to_html(classes="table table-striped", index=False)}
            </div>
            
            <h2 class="mt-4">Visualizations</h2>
            <div class="plot" id="plot1"></div>
            <div class="plot" id="plot2"></div>
            <div class="plot" id="plot3"></div>
            
            <h2 class="mt-4">Sample Outputs</h2>
            <div class="accordion" id="outputAccordion">
    """
    
    # Add sample outputs for each model and question
    for model_name in results_df["model"].unique():
        html_content += f"""
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading{model_name.replace(' ', '')}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                data-bs-target="#collapse{model_name.replace(' ', '')}" aria-expanded="false">
                            {model_name}
                        </button>
                    </h2>
                    <div id="collapse{model_name.replace(' ', '')}" class="accordion-collapse collapse" aria-labelledby="heading{model_name.replace(' ', '')}">
                        <div class="accordion-body">
        """
        
        # Get first iteration results for this model
        model_results = results_df[(results_df["model"] == model_name) & (results_df["iteration"] == 1)]
        
        for _, row in model_results.iterrows():
            html_content += f"""
                            <div class="mb-3">
                                <h5>Question {row['question_idx']}</h5>
                                <p><strong>Input:</strong> {row['question']}</p>
                                <p><strong>Output:</strong> {row['output']}</p>
                                <p><strong>Inference time:</strong> {row['inference_time']:.4f} seconds</p>
                                <hr>
                            </div>
            """
        
        html_content += """
                        </div>
                    </div>
                </div>
        """
    
    # Finish HTML content
    html_content += f"""
            </div>
            
            <h2 class="mt-4">Raw Data</h2>
            <div class="table-responsive">
                {results_df.to_html(classes="table table-striped", index=False)}
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/js/bootstrap.bundle.min.js"></script>
with open("llm_benchmark_results.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print("\nBenchmarking complete. Results saved to llm_benchmark_results.html")

# Also save results as CSV for further analysis
results_df.to_csv("llm_benchmark_results.csv", index=False)
stats_df.to_csv("llm_benchmark_stats.csv", index=False)ayout);
        </script>
    </body>
    </html>
    """
    
    return html_content

# Generate and save HTML report
html_report = generate_html_report(results_df, stats_df)
with open("llm_benchmark_results.html", "w") as f:
    f.write(html_report)

print(f"\nBenchmarking complete. Results saved to llm_benchmark_results.html")

# Also save results as CSV for further analysis
results_df.to_csv("llm_benchmark_results.csv", index=False)
stats_df.to_csv("llm_benchmark_stats.csv", index=False)