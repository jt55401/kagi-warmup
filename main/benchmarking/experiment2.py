from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
)

# System prompt + few-shot examples
system_prompt = """
You are a search query reformulator.
Given a user question, respond ONLY with a JSON object in the following format:

{
  "queries": [
    "search query 1",
    "search query 2"
  ]
}

Do not include any text before or after the JSON.
Do not explain.
Do not answer the question.
Only provide search queries.

Examples:

User: Who lived longer, Nikola Tesla or Milutin Milankovic?
{
  "queries": [
    "Nikola Tesla lifespan",
    "Milutin Milankovic lifespan"
  ]
}

User: In what year was the winner of the 44th edition of the Miss World competition born?
{
  "queries": [
    "44th Miss World competition winner birth year"
  ]
}

User: what are some ways to do fast query reformulation
{
  "queries": [
    "fast query reformulation techniques",
    "query reformulation algorithms",
    "query expansion methods",
    "query rewriting approaches",
    "query refinement strategies"
  ]
}
""".strip()

def generate_queries(user_input: str):
    prompt = f"{system_prompt}\n\nUser: {user_input.strip()}\n"
    
    
    
    prompt = prompt = """
    You are a search query reformulator.
    Respond with a JSON object with key "queries" containing a list of search queries that answer the input.

    Input: What is the capital of France?
    Output:
    {
    "queries": [
        "capital of France"
    ]
    }
    """.strip()
    
    print("\n\nPrompt:", prompt)
    
    result = query_pipeline(prompt)[0]['generated_text']

    try:
        start = result.find('{')
        end = result.rfind('}') + 1
        output = json.loads(result[start:end])
    except json.JSONDecodeError:
        output = {"queries": []}

    return output

# Example
#query = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
query = "What is the capital of France?"
output = generate_queries(query)

print(json.dumps(output, indent=2))
