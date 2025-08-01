from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-medium-persian")
model = AutoModelForCausalLM.from_pretrained("flax-community/gpt2-medium-persian")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    out = generator(prompt, max_length=300, do_sample=True, top_p=0.95, top_k=40)
    return jsonify({"generated_text": out[0]["generated_text"]})