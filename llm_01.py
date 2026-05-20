from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


text = "I love Python and transformers!"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

encoded = tokenizer(text)
print("Encoded Output:", encoded)

decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)