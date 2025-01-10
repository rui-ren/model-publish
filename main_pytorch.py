from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse


app = FastAPI()

torch.random.manual_seed(0)

# Initialize the model, tokenizer, and tokenizer stream
# Please update the DML model here
# model = og.Model('/home/RAI-SDK/hf_version')
# tokenizer = og.Tokenizer(model)
model = AutoModelForCausalLM.from_pretrained(
    "/baseline_model", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)

tokenizer = AutoTokenizer.from_pretrained("/baseline_model")
#tokenizer_stream = tokenizer.create_stream()

print("The model is loaded!!")

# Set the search options
search_options = {
    'max_length': 2048
}

# Define the input schema
class InputData(BaseModel):
    text: str

@app.post("/score")
async def score(input_data: InputData):
    text = input_data.text
    if not text:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # The chat template needs to update later.
    # I will give you the OpenAIChatCompletion template.
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    prompt = chat_template.format(input=text)

    #input_tokens = tokenizer.encode(prompt)

    # params = og.GeneratorParams(model)
    # params.set_search_options(**search_options)
    # params.input_ids = input_tokens
    # generator = og.Generator(model, params)

    # output_text = ""
    # try:
    #     while not generator.is_done():
    #         generator.compute_logits()
    #         generator.generate_next_token()

    #         new_token = generator.get_next_tokens()[0]
    #         output_text += tokenizer_stream.decode(new_token)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error during generation: {e}")
    # finally:
    #     del generator

   

    messages = [
        #{"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": text},
        #{"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        #{"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output_text = pipe(messages, **generation_args)[0]['generated_text']

    return {"response": output_text}

#uvicorn main:app --reload 