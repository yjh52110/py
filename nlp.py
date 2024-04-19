from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from peft import PeftModel


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("do_POST")
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        # Parse JSON data
        json_data = json.loads(post_data)
        input_texts = json_data.get("text", [])

        print(input_texts)
        # Format input texts into prompts
        prompts = []
        for input_text in input_texts:
            prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
            Input: {input_text}
            Answer: """
            prompts.append(prompt)

        # Generate results
        results = []
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, max_length=512)
        res = model.generate(**tokens, max_length=512)
        res_sentences = [tokenizer.decode(i) for i in res]
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        results.extend(out_text)

        # Prepare response
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response_data = json.dumps({"sentiments": results})
        self.wfile.write(response_data.encode("utf-8"))


def load_models():
    base_model = "NousResearch/Llama-2-13b-hf"
    peft_model = "FinGPT/fingpt-sentiment_llama2-13b_lora"
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_models()
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Starting server on port 8000...")
    httpd.serve_forever()

