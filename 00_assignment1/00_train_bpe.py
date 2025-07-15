from pathlib import Path
from collections import Counter
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from typing import BinaryIO
import re
import os
import time
import json
import psutil

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def remove_special_tokens(text: str,
                          special_tokens: list[str]) -> list[str]:
    """
    Remove special tokens from the text.
    """
    pattern = "|".join(map(re.escape, special_tokens))  # Safely escape special characters
    chunk = re.split(pattern, text)
    if chunk[0] == "":  # Skip the first empty chunk if it exists
        chunk = chunk[1:]
    return chunk

def get_chunks_from_file(file_path,
                         num_processes,
                         special_token) -> list[list[str]]:
    """
    Reads a file, finds chunk boundaries using a special token, and returns a list of text chunks.
    """
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_token.encode("utf-8"))
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(remove_special_tokens(text=chunk, special_tokens=[special_token]))
    return chunks

def calculate_word_frequency(chunk):
    word_freq = Counter()
    for sentence in chunk:
        words = str(sentence).split()
        word_freq.update(words)
    return word_freq

def extract_unique_chars(chunk):
    unique_chars = set()
    for word in chunk:
        for char in word:
            unique_chars.add(char)
    return unique_chars

def calculate_bigrams(symbol_sequence_chunk) -> Counter:
    bigram_freq = Counter()
    for symbol_sequence in symbol_sequence_chunk:
        for symbols in symbol_sequence:
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                bigram_freq[pair] += 1
    return bigram_freq

def initialize_symbol_sequences(chunk):
    symbol_sequences = [[] for _ in range(len(chunk))]
    for i, story in enumerate(chunk):
        for word in story.split():
            symbol_sequences[i].append(list(word))
    return symbol_sequences

def merge_pair_single(symbols_list, most_freq_pair, new_symbol):
    new_symbols = []
    for symbols in symbols_list:
        new_seq = []
        i = 0
        while i < len(symbols)-1:
            if(symbols[i], symbols[i+1]) == most_freq_pair:
                new_seq.append(new_symbol)  # merge
                i += 2
            else:
                new_seq.append(symbols[i])
                i += 1
        # if len(symbols) != len(new_seq):
            # print(f"before -> after: {len(symbols), len(new_seq)}")
        if new_seq != []:
            new_symbols.append(new_seq)
    # print(len(new_symbols))
    return new_symbols

def train_bpe(input_path:str, # Path to a text file with BPE tokenizer training data
              vocab_size:int, # A positive integer that defines the maximum final vocabulary size
              special_tokens:list[str], # A list of special tokens to be removed from the text
              num_processes:int,
              temp_path:Path):
    if(num_processes <= 0):
        num_processes = 1
        print(f"Number of processes: {num_processes}")

    # Load the file and get chunks and remove special tokens
    chunks:list[list[str]] = get_chunks_from_file(file_path = input_path,
                                                  num_processes = num_processes,
                                                  special_token = special_tokens[0])
    
    # Initialize the vocabulary and merges
    vocab:dict[int, bytes] = {}
    merges:list[tuple[bytes, bytes]] = []

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')  # Add the special token to the vocabulary

    # Parallel extraction of unique characters
    print("Extracting unique characters from chunks in parallel...")
    with Pool(processes=num_processes) as pool:
        unique_chars_list = []
        for chars in tqdm(pool.imap(extract_unique_chars, chunks), total=len(chunks)):
            unique_chars_list.append(chars)

    # Merge all unique characters from each chunk
    all_unique_chars = set()
    for chars in unique_chars_list:
        for char in chars:
            all_unique_chars.update(char)

    # Build the vocabulary from all unique characters
    for char in all_unique_chars:
        vocab[len(vocab)] = char.encode("utf-8")  # Store characters as bytes
    print("Unique characters extracted")

    # print("Calculating word frequencies in parallel...")
    # with Pool(processes=num_processes) as pool:
    #     results = []
    #     for res in tqdm(pool.imap(calculate_word_frequency, chunks), total=len(chunks)):
    #         results.append(res)

    # # Combine results from all processes
    # total_word_freq = Counter()
    # for word_freq in results:
    #     total_word_freq.update(word_freq)
    # print("Word frequencies calculated.")

    # Initialize symbol sequences with parallel processing
    print("Initializing symbol sequences in parallel...")
    symbol_sequences = []
    # with Pool(processes=num_processes) as pool:
    #     for seq in tqdm(pool.imap(initialize_symbol_sequences, (chunk for chunk in chunks)), total=len(chunks)):
    #         symbol_sequences.append(seq)
    for i,chunk in tqdm(enumerate(chunks)):
        symbol_sequence = initialize_symbol_sequences(chunk)
        save_path = temp_path / f'symbol_sequence_{i}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(symbol_sequence, f)

    # Flatten the list of lists

    print("Symbol sequences initialized.")

    return vocab, merges
    while len(vocab) < vocab_size:
        # 1. Count bigrams over all words in all stories
        start_time = time.time()
        with Pool(processes=num_processes) as pool:
            bigram_freq_list = pool.map(calculate_bigrams, (symbol_sequence for symbol_sequence in symbol_sequences))
        bigram_freq = Counter()
        for item in bigram_freq_list:
            bigram_freq += item
        if not bigram_freq:
            break

        # 2. Find most frequent bigram
        most_freq_pair = max(bigram_freq, key=bigram_freq.get)
        new_symbol = ''.join(most_freq_pair)
        merges.append(most_freq_pair)
        vocab[new_symbol.encode('utf-8')] = len(vocab)

        # 3. Merge the most frequent pair in all words of all stories
        for i in range(len(symbol_sequences)):
            with Pool(processes=num_processes) as pool:
                symbol_sequences[i] = pool.starmap(merge_pair_single, [(word, most_freq_pair, new_symbol) for word in symbol_sequences[i]])

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Merged: {most_freq_pair} -> {new_symbol}")
        # print("Current sequences:", symbol_sequences)
        print("Current merges:", merges)
        print("Current vocab:", vocab)
        print(f"Elapsed time for this merge: {elapsed_time:.4f} seconds")
        print("-" * 40)
    return vocab, merges

if __name__ == "__main__":
    FILE_PATH = Path('data/course_01')
    TEMP_PATH = Path('00_self_data/temp')
    if not TEMP_PATH.exists():
        TEMP_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_PATH = FILE_PATH / "TinyStoriesV2-GPT4-train.txt"
    VALID_DATA_PATH = FILE_PATH / "TinyStoriesV2-GPT4-valid.txt"
    # NUM_PROCESSES = psutil.cpu_count(logical=False)  # Number of processes to use for parallel processing
    NUM_PROCESSES = 8  # Set to a fixed number for testing

    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path=TRAIN_DATA_PATH,
                              vocab_size=10000,
                              special_tokens=special_tokens,
                              num_processes=NUM_PROCESSES,
                              temp_path=TEMP_PATH)
    
    # Ensure the output directory exists
    output_dir = Path("00_self_data/tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open("00_self_data/tokenizer/valid_vocab.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v.decode("utf-8") if isinstance(v, bytes) else v for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)

    with open("00_self_data/tokenizer/valid_merges.json", "w", encoding="utf-8") as f:
        json.dump([(a.decode("utf-8") if isinstance(a, bytes) else a, b.decode("utf-8") if isinstance(b, bytes) else b) for a, b in merges], f, ensure_ascii=False, indent=2)

    print("Vocab and merges saved to valid_vocab.json and valid_merges.json")