import regex as re
from pathlib import Path
import pickle
from typing import Iterable, Iterator
import json

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab:dict[int, bytes] = vocab
        self.merges:list[tuple[bytes, bytes]] = merges
        if (special_tokens):
            special_tokens.sort(key=lambda x: len(x), reverse=True)
        self.special_tokens:list[str]|None = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        if (".pkl" in vocab_filepath.name):
            with open(vocab_filepath, "rb") as vf:
                vocab = pickle.load(vf)
        if (".pkl" in merges_filepath.name):
            with open(merges_filepath, "rb") as mf:
                merges = pickle.load(mf)
        if (".json" in vocab_filepath.name):
            with open(vocab_filepath, "r", encoding="utf-8") as vf:
                vocab = json.load(vf)
        if (".txt" in merges_filepath.name):
            with open(merges_filepath, "r", encoding="utf-8") as mf:
                merges = [tuple(line.strip().split()) for line in mf.readlines()]
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text:str) -> list[int]:
        """
        Encodes the input text into a list of token IDs.
        """
        # step 0. pre-tokenize the text
        def pre_tokenize(text, pattern, special_tokens=None):
            pre_tokens = []
            specials = special_tokens or []
            i = 0
            while i < len(text):
                # Check for any special token at the current position
                matched_special = None
                for special in specials:
                    if text.startswith(special, i):
                        matched_special = special
                        break
                if matched_special:
                    pre_tokens.append((matched_special.encode("utf-8"),))
                    i += len(matched_special)
                    continue
                # Find the next special token position
                next_special_pos = len(text)
                for special in specials:
                    pos = text.find(special, i)
                    if pos != -1 and pos < next_special_pos:
                        next_special_pos = pos
                # Apply regex to the region before the next special token
                region = text[i:next_special_pos]
                for match in re.finditer(pattern, region):
                    word = match.group(0)
                    bytes_tuple = tuple(bytes([b]) for b in word.encode("utf-8", errors="replace"))
                    pre_tokens.append(bytes_tuple)
                i = next_special_pos
            return pre_tokens

        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = pre_tokenize(text, pattern, self.special_tokens)

        # step 1. merge pre-tokens
        for i in range(len(self.merges)):
            pair = self.merges[i]
            new_pre_tokens = []
            for token in pre_tokens:
                if len(token) < 2:
                    new_pre_tokens.append(token)
                    continue
                merged = []
                j = 0
                while j < len(token):
                    if j < len(token) - 1 and (token[j], token[j+1]) == pair:
                        merged.append(token[j] + token[j+1])
                        j += 2
                    else:
                        merged.append(token[j])
                        j += 1
                new_pre_tokens.append(tuple(merged))
            pre_tokens = new_pre_tokens

        # step 2. convert pre-tokens to token IDs
        token_ids = []
        for token in pre_tokens:
            for byte in token:
                for token_id in range(0, len(self.vocab)):
                    if self.vocab[token_id] == byte:
                        token_ids.append(token_id)
                        break
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
        
    def decode(self, token_ids:list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        output_string = b""
        for token_id in token_ids:
            if token_id in self.vocab:
                output_string += self.vocab[token_id]
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary.")
        return output_string.decode("utf-8", errors="replace")

# Vocabulary and Merges location
# location = Path("E:/Data/OneDrive/Data/Code/Python/Language_Modeling_From_Scratch/results")

# vocab_file = location / "tiny_stories_vocab.pkl"
# merges_file = location / "tiny_stories_merges.pkl"

# vocab_file = location / "gpt2_vocab.json"
# merges_file = location / "gpt2_merges.txt"

# pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]

# tokenizer = Tokenizer.from_files(vocab_filepath=vocab_file, merges_filepath=merges_file)
# tokenizer = Tokenizer.from_files(vocab_filepath=vocab_file, merges_filepath=merges_file, special_tokens=special_tokens)

# test_text = (
#     "In the heart of the bustling city, skyscrapers reached for the clouds, their glass windows reflecting the golden rays of the setting sun. "
#     "People hurried along the sidewalks, weaving between street vendors selling fragrant food and artists painting vibrant murals on brick walls. "
#     "At the city park, children laughed as they chased pigeons, while joggers circled the pond and old friends shared stories on weathered benches. "
#     "As night fell, neon lights flickered to life, illuminating the avenues with a rainbow of colors. "
#     "A gentle breeze carried the distant sound of music from an open-air concert, blending with the hum of traffic and the occasional bark of a dog. "
#     "In a quiet apartment above a bakery, a young writer sat by the window, inspired by the city’s energy and dreaming of stories yet to be told. "
#     "With every keystroke, the world outside became a little more magical, and the city’s heartbeat echoed in every word."
# )

# test_text = "你好🙃hello world "
# test_text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
# test_text = "hello world"

# test_text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

# print(test_text.startswith("<|endoftext|>", 10))

# encoded_ids = tokenizer.encode(test_text)
# tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]

# print(encoded_ids)
# print(tokenized_string)

# encoded_tokens = tokenizer.encode(test_text)
# decoded_text = tokenizer.decode(encoded_tokens)

# print(f"Encoded Tokens: {encoded_tokens}")
# print(f"Decoded Text: {decoded_text}")
