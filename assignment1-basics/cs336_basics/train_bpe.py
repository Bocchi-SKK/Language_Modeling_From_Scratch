import regex as re
from typing import Tuple
from collections import defaultdict
from time import time
from pathlib import Path
import os
import pickle
from tqdm import trange, tqdm
import math

def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]
):
    def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    def update_pre_tokens_count(pre_tokens_count, pair_to_merge):
        new_pre_tokens_count = defaultdict(int)
        for token, count in pre_tokens_count.items():
            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == pair_to_merge:
                    merged.append(token[i] + token[i+1])  # merge bytes
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            new_token = tuple(merged)
            new_pre_tokens_count[new_token] += count
        return new_pre_tokens_count

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab:dict[int, bytes] = {}
    merges:list[tuple[bytes, bytes]] = []
    pre_tokens_count = defaultdict(int)

    # Vocabulary initialization
    for i in range(0, 256):
        vocab[len(vocab)] = bytes([i])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    delimiter = "|".join([re.escape(token) for token in special_tokens])
    chunks = re.split(delimiter, text)
    
    for chunk in chunks:
        for word in re.finditer(PAT, chunk):
            token = to_bytes_tuple(word.group(0))
            pre_tokens_count[token] += 1

    # # For large files, we will read the file in parts to avoid memory issues
    # num_parts = 4  # Split the input into 4 parts
    # with open(input_path, "r", encoding="utf-8") as f:
    #     lines = f.readlines()

    # total_lines = len(lines)
    # lines_per_part = math.ceil(total_lines / num_parts)

    # for part in range(num_parts):
    #     start = part * lines_per_part
    #     end = min((part + 1) * lines_per_part, total_lines)
    #     text = "".join(lines[start:end])
    #     delimiter = "|".join([re.escape(token) for token in special_tokens])
    #     chunks = re.split(delimiter, text)
    #     for chunk in tqdm(chunks, desc="Processing chunks"):
    #         for word in re.finditer(PAT, chunk):
    #             token = to_bytes_tuple(word.group(0))
    #             pre_tokens_count[token] += 1

    del chunks  # Free memory
    pass

    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0
    # while len(vocab) < vocab_size:
    for _ in trange(len(vocab), vocab_size, desc="BPE merges"):
        start_time = time()
        # Find the most common pair of tokens
        pairs = defaultdict(int)
        for token, count in pre_tokens_count.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pairs[pair] += count

        if not pairs:
            break
        end_time = time()
        time0 += end_time - start_time

        start_time = time()
        # Get the most common pair
        max_count = max(pairs.values())
        pair_to_merge = []
        # pair_to_merge = [pair for pair, count in pairs.items() if count == max_count]
        for pair, count in pairs.items():
            if count == max_count:
                pair_to_merge.append(pair)
        pair_to_merge = max(pair_to_merge)
        end_time = time()
        time1 += end_time - start_time

        start_time = time()
        # Update the pre_tokens_count with the merged pair
        pre_tokens_count = update_pre_tokens_count(pre_tokens_count, pair_to_merge)
        end_time = time()
        time2 += end_time - start_time

        start_time = time()
        # Update the vocabulary with the merged pair and add the merge operation
        new_token = pair_to_merge[0] + pair_to_merge[1]
        vocab[len(vocab)] = new_token
        merges.append((pair_to_merge[0], pair_to_merge[1]))
        end_time = time()
        time3 += end_time - start_time

    print(f"Time for finding pairs: {time0:.4f} seconds")
    print(f"Time for getting max pair: {time1:.4f} seconds")
    print(f"Time for updating pre_tokens_count: {time2:.4f} seconds")
    print(f"Time for updating vocabulary and merges: {time3:.4f} seconds")
    return vocab, merges

if __name__ == "__main__":

    DATA_PATH = Path("D:/Data/Language_Modeling_From_Scratch/data")
    SAVE_PATH = Path("E:/Data/OneDrive/Data/Code/Python/Language_Modeling_From_Scratch/results")

    if not DATA_PATH.exists():
        os.makedirs(DATA_PATH)
    if not SAVE_PATH.exists():
        os.makedirs(SAVE_PATH)

    VALID_DATA_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    TRAIN_DATA_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    TINY_DATA_PATH = DATA_PATH / "corpus.en"
    OPEN_WEB_TEXT_TRAIN_DATA_PATH = DATA_PATH / "owt_train.txt"
    OPEN_WEB_TEXT_VALID_DATA_PATH = DATA_PATH / "owt_valid.txt"

    start_time = time()
    vocab, merges = train_bpe(
        input_path=TINY_DATA_PATH,
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    end_time = time()
    print(f"Training completed in {end_time - start_time:.4f} seconds")

    # # Save the vocabulary and merges
    # with open(SAVE_PATH / "open_web_text_vocab.txt", "w", encoding="utf-8") as f:
    #     for idx, token in vocab.items():
    #         f.write(f"{idx}\t{token}\n")
    
    # # Save the merges
    # with open(SAVE_PATH / "open_web_text_merges.txt", "w", encoding="utf-8") as f:
    #     for pair in merges:
    #         f.write(f"{pair[0]} {pair[1]}\n")

    # # Save vocab as pickle
    # with open(SAVE_PATH / "open_web_text_vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)

    # # Save merges as pickle
    # with open(SAVE_PATH / "open_web_text_merges.pkl", "wb") as f:
    #     pickle.dump(merges, f)