FROM_REMOTE=True

base_model = 'llama2'
peft_model = 'FinGPT/fingpt-mt_llama2-7b_lora' if FROM_REMOTE else 'finetuned_models/MT-llama2-linear_202309210126'

model, tokenizer = load_model(base_model, peft_model, FROM_REMOTE)


demo_tasks = [
    'Financial Sentiment Analysis',
    'Financial Relation Extraction',
    'Financial Headline Classification',
    'Financial Named Entity Recognition',
]
demo_inputs = [
    "Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano",
    "Apple Inc Chief Executive Steve Jobs sought to soothe investor concerns about his health on Monday, saying his weight loss was caused by a hormone imbalance that is relatively simple to treat.",
    'gold trades in red in early trade; eyes near-term range at rs 28,300-28,600',
    'This LOAN AND SECURITY AGREEMENT dated January 27 , 1999 , between SILICON VALLEY BANK (" Bank "), a California - chartered bank with its principal place of business at 3003 Tasman Drive , Santa Clara , California 95054 with a loan production office located at 40 William St ., Ste .',
]
demo_instructions = [
    'What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.',
    'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.',
    'Does the news headline talk about price going up? Please choose an answer from {Yes/No}.',
    'Please extract entities and their types from the input sentence, entity types should be chosen from {person/organization/location}.',
]

test_demo(model, tokenizer)






