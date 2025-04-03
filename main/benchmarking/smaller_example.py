import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "prhegde/t5-query-reformulation-RL"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()



questions = [
    "In what year was the winner of the 44th edition of the Miss World competition born?",
    "Who lived longer, Nikola Tesla or Milutin Milankovic?",
    "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?",
    "Create a table for top noise cancelling headphones that are not expensive",
    "what are some ways to do fast query reformulation",
]

for question in questions:
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    print(f'\n\nInput: {question}')

    nsent = 4
    with torch.no_grad():
        for i in range(nsent):
            output = model.generate(input_ids, max_length=35, num_beams=1, do_sample=True, repetition_penalty=1.8)
            target_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'Target: {target_sequence}')
