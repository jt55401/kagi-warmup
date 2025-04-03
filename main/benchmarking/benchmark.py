"""
LLM Benchmarking Script for comparing different models.

This script benchmarks various language models using the same set of questions,
tracking both load time and inference time for each model.
"""
import gc
import json
import os
import time
from typing import Dict, List

import pandas as pd
import torch
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


def format_prompt(model_info, question):
    """Format the prompt according to the model type."""
    model_type = model_info["type"]
    model_name = model_info["name"]
    
    system = "Please take the following question and reformulate it to 3 different search queries. Keep the queries short, using minimal meaningful keywords. Respond with the reformulated queries only. Do not answer the question posed. Respond with each query on it's own line."
    
    if model_type == "causal":
        if "Phi-3" in model_name:
            return f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>"
        elif "Llama" in model_name and "Instruct" in model_name:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif "Gemma" in model_name:
            return f"<start_of_turn>system\n{system}<end_of_turn>\n<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model"
        else:
            # Default for other causal models
            return f"System: {system}\nQuestion: {question}\nAnswer:"
    elif model_type == "seq2seq":
        if "query-reformulation" in model_name:
            return f"system: {system}\nreformulate: {question}"
        else:
            return f"system: {system}\nAnswer this question: {question}"


def benchmark_model(model_info, questions, iterations=10):
    """Benchmark a model with given questions for multiple iterations."""
    model_name = model_info["name"]
    model_id = model_info["id"]
    model_type = model_info["type"]
    requires_token = model_info.get("token", False)
    
    print(f"\nBenchmarking {model_name}...")
    
    results = []
    
    for i in range(iterations):
        clear_gpu_memory()
        
        # Load model and measure loading time
        start_time = time.time()
        
        if model_type == "causal":
            if model_id == "distilgpt2":
                tokenizer = GPT2Tokenizer.from_pretrained(model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load with appropriate parameters
            load_params = {"torch_dtype": torch.float16}
            if requires_token:
                load_params["use_auth_token"] = True
            
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_params)
            
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            
        elif model_type == "seq2seq":
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
        
        loading_time = time.time() - start_time
        
        # Process each question
        for q_idx, question in enumerate(questions):
            formatted_prompt = format_prompt(model_info, question)
            reformulations = []
            
            # Get 3 different responses/reformulations
            for k in range(3):
                start_time = time.time()
                
                if model_type == "causal":
                    output = pipe(
                        formatted_prompt, 
                        max_new_tokens=100, 
                        do_sample=(k > 0),  # First one deterministic, others with sampling
                        temperature=0.7 if k > 0 else 1.0,
                        top_p=0.95
                    )[0]["generated_text"]
                    
                    # For causal models, extract generated text
                    if output.startswith(formatted_prompt):
                        output = output[len(formatted_prompt):].strip()
                else:
                    output = pipe(
                        formatted_prompt, 
                        max_length=100,
                        do_sample=(k > 0),  # First one deterministic, others with sampling
                        temperature=0.7 if k > 0 else 1.0,
                        top_p=0.95
                    )[0]["generated_text"]
                
                inference_time = time.time() - start_time
                reformulations.append(output)
                
                # Only record full details for the first reformulation
                if k == 0:
                    results.append({
                        "model": model_name,
                        "iteration": i + 1,
                        "question_idx": q_idx + 1,
                        "question": question,
                        "output": output,
                        "all_reformulations": reformulations,
                        "inference_time": inference_time,
                        "loading_time": loading_time if q_idx == 0 else None  # Only record loading time once per iteration
                    })
                
            # Update all reformulations for the final result
            results[-1]["all_reformulations"] = reformulations
        
        # Explicitly delete model and tokenizer
        del model, tokenizer, pipe
        clear_gpu_memory()
    
    return results

models = [
    {"name": "Phi-3-mini", "id": "microsoft/phi-3-mini-4k-instruct", "type": "causal"},
    {"name": "Llama3.2-1B", "id": "meta-llama/Llama-3.2-1B", "type": "causal", "token": True},
    {"name": "Llama3.2-3B", "id": "meta-llama/Llama-3.2-3B", "type": "causal", "token": True},
    {"name": "Llama3.2-1B-Instruct", "id": "meta-llama/Llama-3.2-1B-Instruct", "type": "causal", "token": True},
    {"name": "Llama3.2-3B-Instruct", "id": "meta-llama/Llama-3.2-3B-Instruct", "type": "causal", "token": True},
    {"name": "Gemma3-1B", "id": "google/gemma-3-1b-it", "type": "causal", "token": True},
    {"name": "Gemma3-4B", "id": "google/gemma-3-4b-it", "type": "causal", "token": True},
    {"name": "DistilGPT2", "id": "distilgpt2", "type": "causal"},
    {"name": "T5-Base", "id": "google/t5-base", "type": "seq2seq"},
    {"name": "T5-Small", "id": "google/t5-small", "type": "seq2seq"},
    {"name": "T5-Tiny", "id": "google/t5-tiny", "type": "seq2seq"},
    {"name": "T5-efficient-Base", "id": "google/t5-efficient-base", "type": "seq2seq"},
    {"name": "T5-efficient-Small", "id": "google/t5-efficient-small", "type": "seq2seq"},
    {"name": "T5-efficient-Tiny", "id": "google/t5-efficient-tiny", "type": "seq2seq"},
    {"name": "T5-query-reformulation-RL", "id": "prhegde/t5-query-reformulation-RL", "type": "seq2seq"}
]

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

# Save results as JSON
with open("llm_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

# Calculate aggregate statistics
stats_df = results_df.groupby("model").agg({
    "inference_time": ["mean", "std", "min", "max"],
    "loading_time": ["mean", "std", "min", "max"]
}).reset_index()

# Flatten the multi-index columns
stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]

# Save stats as JSON
stats_dict = stats_df.to_dict(orient="records")
with open("llm_benchmark_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats_dict, f, indent=2)

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
                                <hr>
                                <h6>Reformulations:</h6>
            """
            
            # Display all reformulations
            for idx, reformulation in enumerate(row['all_reformulations']):
                html_content += f"""
                                <p><strong>Version {idx+1}:</strong> {reformulation}</p>
                """
            
            html_content += f"""
                                <hr>
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
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            const plot1 = {fig1.to_json()};
            const plot2 = {fig2.to_json()};
            const plot3 = {fig3.to_json()};
            
            Plotly.newPlot('plot1', plot1.data, plot1.layout);
            Plotly.newPlot('plot2', plot2.data, plot2.layout);
            Plotly.newPlot('plot3', plot3.data, plot3.layout);
        </script>
    </body>
    </html>
    """
    
    return html_content

# Generate and save HTML report
html_report = generate_html_report(results_df, stats_df)
with open("llm_benchmark_results.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print(f"\nBenchmarking complete. Results saved to llm_benchmark_results.html and JSON files")
