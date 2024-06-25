from KanBERT.kan_bert_tokenizer import KanTokenizer

tokenizer = KanTokenizer()

text = [ 
    'Hello, how are you? I am Romeo.\n',
    'Hello, Romeo My name is Juliet. Nice to meet you.\n',
    'Nice meet you too. How are you today?\n',
    'Great. My baseball team won the competition.\n',
    'Oh Congratulations, Juliet\n',
    'Thanks you Romeo'
]

tokenizer.ingest_vocab_batch(text=text)

x = tokenizer.encode(text[0])
print(x)
y = tokenizer.decode(x)
print(y)