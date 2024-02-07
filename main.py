from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel  # 0.5.0

# Load Models
base_model = "internlm/internlm-20b" 
peft_model = "FinGPT/fingpt-sentiment_internlm-20b_lora"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="cpu", load_in_8bit=True)
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()

# Make prompts
prompt = [
    '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
    Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
    Answer: ''',
    '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
    Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
    Answer: ''',
    '''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
    Input: A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .
    Answer: ''',
]

# Generate results
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]

# show results
for sentiment in out_text:
    print(sentiment)





