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
from filelock import FileLock

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

def calculate_bigrams(temp_path, num_chunk) -> Counter:
    bigram_freq = Counter()
    load_path = temp_path / f'symbol_sequence_{num_chunk}.pkl'
    with open(load_path, 'rb') as f:
        symbol_sequences = pickle.load(f)
    for symbol_sequence in symbol_sequences:
        for symbols in symbol_sequence:
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                bigram_freq.update([pair])
    return bigram_freq

def initialize_symbol_sequences(chunk):
    symbol_sequences = [[] for _ in range(len(chunk))]
    for i, story in enumerate(chunk):
        for word in story.split():
            symbol_sequences[i].append(list(word))
    return symbol_sequences

def merge_pair_single(temp_path, num_chunk, most_freq_pair, new_symbol_input):
    new_symbols_list = []
    load_path = temp_path / f'symbol_sequence_{num_chunk}.pkl'
    lock_path = str(load_path) + ".lock"
    with FileLock(lock_path):
        with open(load_path, 'rb') as f:
            symbols_list = pickle.load(f)   # load stories

        for symbols in symbols_list:  # load one story
            new_symbol = []
            for symbol in symbols:  # load one word
                new_seq = []
                i = 0
                while i < len(symbol):
                    if(i < len(symbol)-1 and (symbol[i], symbol[i+1]) == most_freq_pair):
                        new_seq.append(new_symbol_input)  # merge
                        i += 2
                    else:
                        new_seq.append(symbol[i])
                        i += 1
                new_symbol.append(new_seq) # words
            new_symbols_list.append(new_symbol)

        try:
            test = new_symbols_list[0][0][0]
        except Exception:
            new_symbols_list = [[[]]]  # If empty, return a list with an empty word
        
        with open(load_path, 'wb') as f:
            pickle.dump(new_symbols_list, f)
    return

def train_bpe(input_path:str, # Path to a text file with BPE tokenizer training data
              vocab_size:int, # A positive integer that defines the maximum final vocabulary size
              special_tokens:list[str], # A list of special tokens to be removed from the text
              num_processes:int,
              num_chunks:int,
              temp_path:Path):
    if(num_processes <= 0):
        num_processes = 1
        print(f"Number of processes: {num_processes}")

    # Load the file and get chunks and remove special tokens
    chunks:list[list[str]] = get_chunks_from_file(file_path = input_path,
                                                  num_processes = num_chunks,
                                                  special_token = special_tokens[0])
    
    # Initialize the vocabulary and merges
    vocab:dict[int, bytes] = {}
    merges:list[tuple[bytes, bytes]] = []

    for special_token in special_tokens:
        vocab[special_token.encode('utf-8')] = len(vocab)

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
        vocab[char.encode("utf-8")] = len(vocab)
    print("Unique characters extracted")

    # Initialize symbol sequences
    print("Initializing symbol sequences...")
    for i,chunk in tqdm(enumerate(chunks), total=len(chunks)):
        symbol_sequence = initialize_symbol_sequences(chunk)
        save_path = temp_path / f'symbol_sequence_{i}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(symbol_sequence, f)

    # Flatten the list of lists

    print("Symbol sequences initialized.")

    while len(vocab) < vocab_size:
        # 1. Count bigrams over all words in all stories
        start_time = time.time()
        bigram_freq = Counter()
        with Pool(processes=num_processes) as pool:
            for freq in pool.starmap(calculate_bigrams, [(temp_path, i) for i in range(num_chunks)]):
                bigram_freq.update(freq)

        # 2. Find most frequent bigram
        most_freq_pair = max(bigram_freq, key=bigram_freq.get)
        new_symbol = ''.join(most_freq_pair)
        merges.append(most_freq_pair)
        vocab[new_symbol.encode('utf-8')] = len(vocab)

        # 3. Merge the most frequent pair in all words of all stories
        with Pool(processes=num_processes) as pool:
            pool.starmap(merge_pair_single, [(temp_path, i, most_freq_pair, new_symbol) for i in range(num_chunks)])

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Merged: {most_freq_pair} -> {new_symbol}")
        # print("Current sequences:", symbol_sequences)
        # print("Current merges:", merges)
        # print("Current vocab:", vocab)
        print(f"Current vocabulary size: {len(vocab)}")
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
    TINY_DATA_PATH = FILE_PATH / "TinyStoriesV2-GPT4-tiny.txt"
    # NUM_PROCESSES = psutil.cpu_count(logical=False)  # Number of processes to use for parallel processing
    NUM_PROCESSES = 4  # Set to a fixed number for testing
    NUM_CHUNKS = NUM_PROCESSES*2  # Number of chunks to split the data into

    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path=VALID_DATA_PATH,
                              vocab_size=10000,
                              special_tokens=special_tokens,
                              num_processes=NUM_PROCESSES,
                              num_chunks=NUM_CHUNKS,
                              temp_path=TEMP_PATH)
    
    # Delete all files in TEMP_PATH
    for temp_file in TEMP_PATH.glob("*"):
        if temp_file.is_file():
            temp_file.unlink()

    # Ensure the output directory exists
    output_dir = Path("00_self_data/tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open("00_self_data/tokenizer/valid_vocab.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v.decode("utf-8") if isinstance(v, bytes) else v for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)

    with open("00_self_data/tokenizer/valid_merges.json", "w", encoding="utf-8") as f:
        json.dump([(a.decode("utf-8") if isinstance(a, bytes) else a, b.decode("utf-8") if isinstance(b, bytes) else b) for a, b in merges], f, ensure_ascii=False, indent=2)

    print("Vocab and merges saved to valid_vocab.json and valid_merges.json")