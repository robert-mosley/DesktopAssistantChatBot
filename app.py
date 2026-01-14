from peft import AutoPeftModelForCausalLM
import asyncio
from fastapi import FastAPI
import uvicorn
import os
import nest_asyncio
import socket
import json
from transformers import pipeline, AutoTokenizer
import time
import torch
import requests
app = FastAPI()
model_ask = AutoPeftModelForCausalLM.from_pretrained(
    "robertthecreator/assistant-chatbot",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer_ask = AutoTokenizer.from_pretrained("robertthecreator/assistant-chatbot")
tokenizer_ask.pad_token = tokenizer_ask.eos_token
model_ask.eval()
api_key = os.getenv("TAVILY_API_KEY", "default-key")

def get_from_web(query, k=3):
    context = ""
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={"query": query, "api_key": api_key}
    )
    data = response.json()
    for result in data["results"]:
        print(result["content"])
        context += result["content"] + "\n"
    return context


@app.post("/ask")
def ask(text: str):
    global context
    print("starting...")
    context = """You're an AI assistant designed to help me with every day task, productivity and engineering. Youre name is JARVIS. Use the context you have to answer any questions."""
    context = get_from_web(text) + "\n" + context
    print(context)
    formatted_prompt = f"""CONTEXT: {context}\nPROMPTER: {text}\nASSISTANT:"""
    inputs = tokenizer_ask(formatted_prompt, return_tensors="pt").to(model_ask.device)

    with torch.no_grad():
        outputs = model_ask.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer_ask.eos_token_id,
            eos_token_id=tokenizer_ask.eos_token_id,
            early_stopping=True
        )


    response = tokenizer_ask.decode(outputs[0], skip_special_tokens=True)
    print(response)

    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()

    response = response.split('\r')[0]
    response = response.split('PROMPTER')[0]
    response = response.split('ROLE:')[0]
    response = response.split('<|user|>')[0]
    response = response.split('User:')[0]

    return response

async def start():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)

    await server.serve()

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")