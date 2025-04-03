from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Will use GPU if available
    torch_dtype="auto"
)

# Text generation pipeline
query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    return_full_text=False
)

# Prompt template
TEMPLATE = """
You are a search query reformulator.
Respond with a JSON object with key "queries" containing a list of search queries that answer the input.

Input: {query}
Output:
""".strip()

def generate_search_queries(user_query):
    prompt = TEMPLATE.format(query=user_query.strip())
    response = query_pipeline(prompt)[0]["generated_text"]
    
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        return parsed
    except Exception as e:
        print("⚠️ Failed to parse response:", response)
        return {"queries": []}

# --- Example usage ---
if __name__ == "__main__":
    example = "Who lived longer, Nikola Tesla or Milutin Milankovic?"
    result = generate_search_queries(example)
    print(json.dumps(result, indent=2))
