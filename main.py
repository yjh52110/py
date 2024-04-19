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

# Load Models
base_model = "NousResearch/Llama-2-13b-hf"
peft_model = "FinGPT/fingpt-sentiment_llama2-13b_lora"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(
    base_model, trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
)
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("do_POST method called")
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        input_text = post_data.decode("utf-8")

        # Print received input for debugging
        print("Received input text:", input_text)

        # Generate sentiment analysis
        prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
        Input: {input_text} 
        Answer: """
        tokens = tokenizer(prompt, return_tensors="pt", padding=True, max_length=512)
        res = model.generate(**tokens, max_length=512)
        res_sentences = [tokenizer.decode(i) for i in res]
        out_text = [o.split("Answer: ")[1] for o in res_sentences]

        # Prepare response
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response_data = json.dumps({"sentiment": out_text[0]})
        self.wfile.write(response_data.encode("utf-8"))


def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()

