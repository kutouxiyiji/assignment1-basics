import regex as re
from collections import defaultdict, Counter
import cProfile
import pstats
from io import StringIO
import multiprocessing as mp
from functools import partial
import pickle
import json
from pathlib import Path
from typing import BinaryIO
import os


# '(?:[sdmt]|ll|ve|re) - Contractions: 's, 'd, 'm, 't, 'll, 've, 're
# \p{L}+ or ?\p{L}+ - Letters (with optional leading space)
# \p{N}+ or ?\p{N}+ - Numbers (with optional leading space)
# [^\s\p{L}\p{N}]+ or ?[^\s\p{L}\p{N}]+ - Non-letter/number/space chars (with optional leading space)
# \s+(?!\S) - Whitespace not followed by non-whitespace
# \s+ - Any remaining whitespace
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



class BEP_tokenizer_trainer:

        def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
                # VOCAB = namedtuple("vocab", ["id_to_token", "token_to_id"])
                self.id_to_token = defaultdict()
                self.token_to_id = defaultdict()
                # self.vocab = VOCAB(id_to_token, token_to_id)
                self.vocab_size = vocab_size
                self.special_tokens = special_tokens
                # Init the vocab
                self.id_to_token = {i: bytes([i]) for i in range(256)}
                self.token_to_id = {bytes([i]): i for i in range(256)}
                self.idx = 256
                for special_token in self.special_tokens:
                        b_special_token = special_token.encode("UTF-8")
                        self.id_to_token[self.idx] = b_special_token
                        self.token_to_id[b_special_token] = self.idx
                        self.idx += 1
                if self.idx >= self.vocab_size:
                        raise ValueError(f"The vocab size {vocab_size} is too small to initialize.")
                # other inits
                self.tuple_token_counter = Counter() # dict[tuple[bytes], int]
                escaped_tokens = [re.escape(special_token) for special_token in self.special_tokens]
                self.escaped_pattern = "|".join(escaped_tokens)
                self.merges = [] # list/set of tuple[bytes], e.g. (13, 31)
                self.merges_set = set()
                self.merge_counter = Counter() # bytes: int, bytes = bytes([byte1, byte2]), byte1 and byte 2 are both int
        
        def remove_special_tokens(self, input_text):
                if not self.special_tokens:
                        return input_text
                return re.sub(self.escaped_pattern, "", input_text)

        def _process_chunk(self, text_chunk):
                chunk_counter = Counter()
                for match in re.finditer(PAT, text_chunk):
                        pre_token = match.group()
                        tuple_token = tuple(pre_token.encode("UTF-8"))
                        chunk_counter[tuple_token] += 1
                return chunk_counter

        def pre_tokenizer(self, input_text, use_multiprocessing=True, num_processes=None):
                processed_input_text = self.remove_special_tokens(input_text)
                
                if not use_multiprocessing:
                        for match in re.finditer(PAT, processed_input_text):
                                pre_token = match.group()
                                tuple_token = tuple(pre_token.encode("UTF-8"))
                                self.tuple_token_counter[tuple_token] += 1
                        return
                
                # Multiprocessing implementation
                if num_processes is None:
                        num_processes = mp.cpu_count() - 1 #  try not to freeze the UI by minus 1
                
                # Split text into chunks
                text_length = len(processed_input_text)
                chunk_size = text_length // num_processes
                chunks = []
                
                for i in range(num_processes):
                        start_idx = i * chunk_size
                        if i == num_processes - 1:
                                # Last chunk gets remainder
                                end_idx = text_length
                        else:
                                end_idx = (i + 1) * chunk_size
                                # Extend to next whitespace to avoid splitting tokens
                                while end_idx < text_length and not processed_input_text[end_idx].isspace():
                                        end_idx += 1
                        chunks.append(processed_input_text[start_idx:end_idx])
                
                # Process chunks in parallel
                with mp.Pool(processes=num_processes) as pool:
                        results = pool.map(self._process_chunk, chunks)
                
                # Merge results from all processes
                for chunk_counter in results:
                        for tuple_token, count in chunk_counter.items():
                                self.tuple_token_counter[tuple_token] += count
        
        def init_merge_counter_freq(self, tuple_token):
                if len(tuple_token) <= 1:
                        return 
                for i in range(len(tuple_token) - 1):
                        # bs = bytes([tuple_token[i], tuple_token[i+1]]) # bytes([231, 154])
                        byte1 = self.id_to_token[tuple_token[i]]
                        byte2 = self.id_to_token[tuple_token[i+1]]
                        bs_pair = (byte1, byte2)
                        self.merge_counter[bs_pair] += self.tuple_token_counter[tuple_token]

        def merge_one_tuple_token(self, tuple_token):
                new_tuple_token = []
                if len(tuple_token) <= 1:
                        return tuple_token
                i = 0
                while i < len(tuple_token) - 1:
                        idx1 = tuple_token[i]
                        idx2 = tuple_token[i+1]
                        byte1 = self.id_to_token[idx1]
                        byte2 = self.id_to_token[idx2]
                        # bs = byte1 + byte2
                        bs_pair = (byte1, byte2)
                        # bs = bytes([byte1, byte2])
                        if bs_pair in self.merges_set:
                                bs = bs_pair[0] + bs_pair[1]
                                new_token_id = self.token_to_id[bs]
                                new_tuple_token.append(new_token_id)
                                i += 2
                        else:
                                new_tuple_token.append(idx1)
                                i += 1
                if i == len(tuple_token) - 1:
                        new_tuple_token.append(tuple_token[i])
                return tuple(new_tuple_token)

        def update_merge_counter_after_merge(self, old_tuple_token, new_tuple_token, freq):
                old_pairs = []
                for i in range(len(old_tuple_token) - 1):
                        byte1 = self.id_to_token[old_tuple_token[i]]
                        byte2 = self.id_to_token[old_tuple_token[i + 1]]
                        # old_bs = byte1 + byte2
                        old_bs_pair = (byte1, byte2)
                        # old_bs = bytes([byte1, byte2])
                        old_pairs.append(old_bs_pair)
                new_pairs = []
                for i in range(len(new_tuple_token) - 1):
                        byte1 = self.id_to_token[new_tuple_token[i]]
                        byte2 = self.id_to_token[new_tuple_token[i + 1]]
                        # new_bs = byte1 + byte2
                        new_bs_pair = (byte1, byte2)
                        # new_bs = bytes([byte1, byte2])
                        new_pairs.append(new_bs_pair)
                for pair in old_pairs:
                        self.merge_counter[pair] -= freq
                        if self.merge_counter[pair] <= 0:
                                del self.merge_counter[pair]
                for pair in new_pairs:
                        self.merge_counter[pair] += freq

        # The main merge.
        def update_merge_counter_freq(self, tuple_token): # tuple token is tuple of token ids, i.e. it's a sub-word inited by pre-tokenizer
                if len(tuple_token) <= 1:
                        return
                new_tuple_token = self.merge_one_tuple_token(tuple_token)
                if len(new_tuple_token) == len(tuple_token):
                        return
                # merge happends, update the byte-wise token counter, udpate the sub-word (pre-token) counter
                # print(f"the old tuple token is {tuple_token} and the new tuple token is {new_tuple_token}")
                freq = self.tuple_token_counter[tuple_token]
                self.update_merge_counter_after_merge(tuple_token, new_tuple_token, freq)
                # self.tuple_token_counter[new_tuple_token] = freq
                # Aggregate frequencies when multiple original tokens merge into the same new token
                self.tuple_token_counter[new_tuple_token] += freq
                del self.tuple_token_counter[tuple_token]
        
        # typical step of getting one most freq BP
        # 1. scan token_tuple in self.tuple_token_counter
        # 2. update the self.merge_counter for each token_tuple
        # 3. find the most freq pair in merge_counter
        # 4. update the update the merges
        def step(self):
                keys = sorted(list(self.tuple_token_counter.keys()))
                # TODO: this for loop can become a multi-thread and improve the speed? break the tuple_token into threads and excute the update_merge_counter_freq
                # lock when needed.
                for tuple_token in keys:
                        self.update_merge_counter_freq(tuple_token)
                
                if not self.merge_counter:
                        return
                        
                # find the max freq pair
                # Use max() for O(n) complexity with lexicographic tie-breaking
                # When frequencies are equal, prefer the lexicographically GREATER pair (per spec)
                # key returns (frequency, pair) tuple - max compares lexicographically
                # def sort(item):
                #         bytes, freq = item
                #         return (freq, bytes)
                new_bytes_token_pair = max(self.merge_counter.items(), key=lambda item: (item[1], item[0]))[0]
                # new_bytes_token_pair = max(self.merge_counter.items(), key=lambda item: (item[1], item[0]))[0]
                self.merges.append(new_bytes_token_pair)
                self.merges_set.add(new_bytes_token_pair)
                del self.merge_counter[new_bytes_token_pair]
                # insert the new bytes
                new_bytes_token = new_bytes_token_pair[0] + new_bytes_token_pair[1]
                self.id_to_token[self.idx] = new_bytes_token
                self.token_to_id[new_bytes_token] = self.idx
                self.idx += 1
        
        def train(self, input_file_path, use_multiprocessing=True, num_processes=None, check_profile = False):
                """Train the BPE tokenizer."""
                if check_profile:
                        profiler = cProfile.Profile()
                        
                        # file reading
                        profiler.enable()
                try:
                        with open(input_file_path, 'r', encoding='utf-8') as f:
                                input_text = f.read()
                except UnicodeDecodeError:
                        raise NotImplementedError
                if check_profile:
                        profiler.disable()
                        self._print_profile_stats(profiler, "File Reading")
                
                # pre-tokenization
                if check_profile:
                        profiler = cProfile.Profile()
                        profiler.enable()
                self.pre_tokenizer(input_text, use_multiprocessing=use_multiprocessing, num_processes=num_processes)
                if check_profile:
                        profiler.disable()
                        mode = "Multiprocessing" if use_multiprocessing else "Single-threaded"
                        self._print_profile_stats(profiler, f"Pre-tokenization ({mode})")
                
                # merge counter initialization
                if check_profile:
                        profiler = cProfile.Profile()
                        profiler.enable()
                # Sort keys for deterministic ordering
                for tuple_token in sorted(self.tuple_token_counter.keys()):
                        self.init_merge_counter_freq(tuple_token)
                if check_profile:
                        profiler.disable()
                        self._print_profile_stats(profiler, "Merge Counter Initialization")
                
                # BPE steps
                # pre-tokenizer update the tuple_token_counter which is the main data used later in training step
                if check_profile:
                        profiler = cProfile.Profile()
                        profiler.enable()
                step_count = 0
                while self.idx < self.vocab_size:
                        if self.idx % 1000 == 0:
                                print(f"with total {self.idx} number of vacob, the most recent new vacob is {self.id_to_token[self.idx-1]}")
                        self.step()
                        step_count += 1
                if check_profile:
                        profiler.disable()
                        self._print_profile_stats(profiler, f"BPE Training ({step_count} steps)")
                
                        print(f"We get total {self.idx} number of vacobs. And the merge set is {self.merges}.")
                
                # # Convert merges from (bytes, bytes) tuples to list for return
                # # The internal representation uses (bytes, bytes) pairs
                # merges_as_tuples = []
                # for m in self.merges:
                #         if isinstance(m, tuple) and len(m) == 2:
                #                 # m is already (bytes, bytes)
                                
                #                 merges_as_tuples.append(bytes(list(m)))
                #         else:
                #                 # Fallback for unexpected format
                #                 merges_as_tuples.append(m)
                
                return self.merges, self.id_to_token
        
        def save(self, output_dir):
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save vocab as pickle (handles bytes keys/values efficiently)
                vocab_data = {
                        'id_to_token': dict(self.id_to_token),
                        'token_to_id': dict(self.token_to_id),
                        'vocab_size': self.vocab_size,
                        'idx': self.idx,
                        'special_tokens': self.special_tokens
                }
                
                with open(output_path / 'vocab.pkl', 'wb') as f:
                        pickle.dump(vocab_data, f)
                
                # Save merges as pickle (set of bytes objects)
                with open(output_path / 'merges.pkl', 'wb') as f:
                        pickle.dump(list(self.merges), f)
                
                # Also save a human-readable version for inspection
                vocab_readable = {}
                for k, v in self.id_to_token.items():
                        if isinstance(v, bytes):
                                vocab_readable[k] = v.decode('utf-8', errors='replace')
                        else:
                                vocab_readable[k] = str(v)
                
                with open(output_path / 'vocab_readable.json', 'w', encoding='utf-8') as f:
                        json.dump(vocab_readable, f, indent=2, ensure_ascii=False)
                
                # Handle merges which are now tuples of (token1, token2)
                # After the recent code changes, merges are tuples where items can be:
                # - integers (byte values 0-255 or token IDs)
                # - bytes objects
                merges_readable = []
                for m in self.merges:
                        if isinstance(m, tuple) and len(m) == 2:
                                item1, item2 = m
                                # Convert to bytes
                                if isinstance(item1, int):
                                        # Could be a byte value or a token ID - check if it's in vocab
                                        item1 = self.id_to_token.get(item1, bytes([item1]) if item1 < 256 else b'?')
                                if isinstance(item2, int):
                                        item2 = self.id_to_token.get(item2, bytes([item2]) if item2 < 256 else b'?')
                                if isinstance(item1, bytes) and isinstance(item2, bytes):
                                        merged = item1 + item2
                                        merges_readable.append(merged.decode('utf-8', errors='replace'))
                                else:
                                        merges_readable.append(f"{item1}+{item2}")
                        elif isinstance(m, bytes):
                                # Old format: already concatenated bytes
                                merges_readable.append(m.decode('utf-8', errors='replace'))
                        else:
                                merges_readable.append(str(m))
                
                with open(output_path / 'merges_readable.json', 'w', encoding='utf-8') as f:
                        json.dump(merges_readable, f, indent=2, ensure_ascii=False)
                
                print(f"\n✓ Tokenizer saved to {output_path}")
                print(f"  - vocab.pkl ({len(self.id_to_token)} tokens)")
                print(f"  - merges.pkl ({len(self.merges)} merges)")
                print(f"  - vocab_readable.json (for inspection)")
                print(f"  - merges_readable.json (for inspection)")
        
        @classmethod
        def from_files(cls, vocab_path, merges_path, special_tokens=None):
                """
                Create a tokenizer from separate vocab and merge files.
                
                Args:
                        vocab_path: Path to vocab file (dict with id_to_token mapping)
                        merges_path: Path to merges file (list of merge rules)
                        special_tokens: List of special tokens (optional, will try to infer from vocab)
                
                Returns:
                        BEP_tokenizer_trainer instance
                
                Example:
                        tokenizer = BEP_tokenizer_trainer.from_files(
                                './gpt2_vocab.json',
                                './gpt2_merges.txt',
                                ['<|endoftext|>']
                        )
                """
                # Load vocab (support both pickle and json)
                vocab_path = Path(vocab_path)
                if vocab_path.suffix == '.pkl':
                        with open(vocab_path, 'rb') as f:
                                vocab_data = pickle.load(f)
                        id_to_token = vocab_data.get('id_to_token', vocab_data)
                        token_to_id = vocab_data.get('token_to_id', {v: k for k, v in id_to_token.items()})
                        vocab_size = vocab_data.get('vocab_size', len(id_to_token))
                        if special_tokens is None:
                                special_tokens = vocab_data.get('special_tokens', [])
                elif vocab_path.suffix == '.json':
                        with open(vocab_path, 'r', encoding='utf-8') as f:
                                vocab_dict = json.load(f)
                        # Convert string keys to int, string values to bytes
                        id_to_token = {int(k): v.encode('utf-8') if isinstance(v, str) else v 
                                       for k, v in vocab_dict.items()}
                        token_to_id = {v: k for k, v in id_to_token.items()}
                        vocab_size = len(id_to_token)
                        if special_tokens is None:
                                special_tokens = []
                else:
                        raise ValueError(f"Unsupported vocab file format: {vocab_path.suffix}")
                
                # Load merges (support both pickle and text)
                merges_path = Path(merges_path)
                if merges_path.suffix == '.pkl':
                        with open(merges_path, 'rb') as f:
                                merges_list = pickle.load(f)
                elif merges_path.suffix == '.txt':
                        merges_list = []
                        with open(merges_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                        line = line.strip()
                                        if line and not line.startswith('#'):
                                                # Support formats like "h e" or "he"
                                                parts = line.split()
                                                if len(parts) == 2:
                                                        merge = (parts[0] + parts[1]).encode('utf-8')
                                                elif len(parts) == 1:
                                                        merge = parts[0].encode('utf-8')
                                                else:
                                                        continue
                                                merges_list.append(merge)
                elif merges_path.suffix == '.json':
                        with open(merges_path, 'r', encoding='utf-8') as f:
                                merge_strings = json.load(f)
                        merges_list = [m.encode('utf-8') if isinstance(m, str) else m 
                                       for m in merge_strings]
                else:
                        raise ValueError(f"Unsupported merges file format: {merges_path.suffix}")
                
                # Create instance using the factory method
                return cls.from_pretrained(id_to_token, token_to_id, merges_list, 
                                          vocab_size, special_tokens)
        
        @classmethod
        def from_pretrained(cls, id_to_token, token_to_id, merges, vocab_size=None, 
                           special_tokens=None):
                """
                Create a tokenizer from pre-trained vocab and merges.
                
                Args:
                        id_to_token: Dict mapping token IDs to bytes
                        token_to_id: Dict mapping bytes to token IDs (optional, will be computed)
                        merges: List or set of merge rules (bytes)
                        vocab_size: Vocabulary size (optional, will be inferred)
                        special_tokens: List of special tokens (optional)
                
                Returns:
                        BEP_tokenizer_trainer instance
                
                Example:
                        id_to_token = {0: b'\\x00', 1: b'\\x01', ..., 256: b'the'}
                        merges = [b'th', b'e', b'the']
                        tokenizer = BEP_tokenizer_trainer.from_pretrained(
                                id_to_token, None, merges, special_tokens=['<|endoftext|>']
                        )
                """
                # Infer values if not provided
                if vocab_size is None:
                        vocab_size = len(id_to_token)
                if special_tokens is None:
                        special_tokens = []
                if token_to_id is None:
                        token_to_id = {v: k for k, v in id_to_token.items()}
                
                # Create a minimal instance (skip normal __init__)
                tokenizer = cls.__new__(cls)  # Create instance without calling __init__
                
                # Set attributes directly
                tokenizer.vocab_size = vocab_size
                tokenizer.special_tokens = special_tokens
                tokenizer.id_to_token = id_to_token
                tokenizer.token_to_id = token_to_id
                tokenizer.idx = len(id_to_token)
                tokenizer.merges = list(merges) if not isinstance(merges, list) else merges
                tokenizer.merges_set = set(merges)
                
                # Initialize other attributes that might be needed
                tokenizer.tuple_token_counter = Counter()
                tokenizer.merge_counter = Counter()
                
                # Set up special token pattern
                if special_tokens:
                        import regex as re
                        escaped_tokens = [re.escape(token) for token in special_tokens]
                        tokenizer.escaped_pattern = "|".join(escaped_tokens)
                else:
                        tokenizer.escaped_pattern = ""
                
                print(f"\n✓ Tokenizer created from pretrained data")
                print(f"  - {len(tokenizer.id_to_token)} tokens")
                print(f"  - {len(tokenizer.merges)} merges")
                print(f"  - {len(tokenizer.special_tokens)} special tokens")
                
                return tokenizer
        
        @classmethod
        def load(cls, input_dir):
                """
                Load a tokenizer from a directory containing vocab.pkl and merges.pkl.
                
                Args:
                        input_dir: Directory containing the tokenizer files
                
                Returns:
                        BEP_tokenizer_trainer instance
                """
                input_path = Path(input_dir)      
                # Load vocab
                with open(input_path / 'vocab.pkl', 'rb') as f:
                        vocab_data = pickle.load(f)
                
                # Load merges
                with open(input_path / 'merges.pkl', 'rb') as f:
                        merges_list = pickle.load(f)
                
                # Use from_pretrained to create the instance
                return cls.from_pretrained(
                        id_to_token=vocab_data['id_to_token'],
                        token_to_id=vocab_data['token_to_id'],
                        merges=merges_list,
                        vocab_size=vocab_data['vocab_size'],
                        special_tokens=vocab_data['special_tokens']
                )
        
        def _print_profile_stats(self, profiler, stage_name):
                """Print profiling statistics for a given stage."""
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(10)  # Print top 10 functions
                
                print(f"\n{'='*60}")
                print(f"Profile for: {stage_name}")
                print('='*60)
                print(s.getvalue())
                print('='*60 + '\n')

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk, may read more or less than 4k bytes.

            # If EOF, this boundary should be at the end of the file. The last read will read empty mini_chunk and it equals to b"".
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

# test
# with open('./data/TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8') as f:
#         test_input_text = f.read()
# print(test_input_text.find('<|endoftext|>'))
# print(test_input_text[242:260])

if __name__ == '__main__':
        import sys
        
        # Check if we should load or train
        if len(sys.argv) > 1 and sys.argv[1] == 'load':
                # Load existing tokenizer
                tokenizer_dir = sys.argv[2] if len(sys.argv) > 2 else './tokenizer_output/test'
                print("\n" + "="*60)
                print(f"LOADING TOKENIZER FROM {tokenizer_dir}")
                print("="*60)
                tokenizer = BEP_tokenizer_trainer.load(tokenizer_dir)
                print(f"id_to_token {tokenizer.id_to_token} \n")
                print(f"token_to_id {tokenizer.token_to_id} \n")
                print(f"merges {tokenizer.merges} \n")
                # print(f"merges_set {tokenizer.merges_set} \n")
                print(f"tuple_token_counter {tokenizer.tuple_token_counter} \n")
                print(f"merge_counter {tokenizer.merge_counter} \n")
                
        else:
                # Train new tokenizer
                print("\n" + "="*60)
                print("TRAINING NEW TOKENIZER")
                print("="*60)
                
                vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
                data_file = sys.argv[2] if len(sys.argv) > 2 else './data/TinyStoriesV2-GPT4-valid.txt'
                output_dir = sys.argv[3] if len(sys.argv) > 3 else './tokenizer_output/test'
                
                print(f"Config:")
                print(f"  - Vocab size: {vocab_size}")
                print(f"  - Data file: {data_file}")
                print(f"  - Output dir: {output_dir}")
                print()
                
                tokenizer = BEP_tokenizer_trainer(vocab_size, ['<|endoftext|>'])
                merges, vocab = tokenizer.train(data_file, use_multiprocessing=True, num_processes=None, check_profile = True)
                
                print(f"\nTraining complete:")
                print(f"  - Vocab size: {tokenizer.vocab_size}")
                print(f"  - Final index: {tokenizer.idx}")
                print(f"  - Num merges: {len(tokenizer.merges)}")
                def sample_dict(my_dict):
                        return dict(list(my_dict.items())[-10:])
                print(f" Sampled tuple token counter is {sample_dict(tokenizer.tuple_token_counter)}\n Sampled id to token is {sample_dict(tokenizer.id_to_token)}\n")
                print(f" Sampled token to id is {sample_dict(tokenizer.token_to_id)}\n Sampled merge counter is {sample_dict(tokenizer.merge_counter)}\n")
                print(f" Sampled merges are {tokenizer.merges[:10]} and {tokenizer.merges[-10:]}\n")
                print(f" returned merges are {merges[:10]} and {merges[-10:]}")
                # Save the tokenizer
                tokenizer.save(output_dir)
                
                if False:
                        # Demonstrate loading
                        print("\n" + "="*60)
                        print("TESTING LOAD FUNCTIONALITY")
                        print("="*60)
                        loaded_tokenizer = BEP_tokenizer_trainer.load(output_dir)
                        
                        # Verify they match
                        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
                        assert loaded_tokenizer.idx == tokenizer.idx
                        assert len(loaded_tokenizer.merges) == len(tokenizer.merges)
                        assert loaded_tokenizer.id_to_token == tokenizer.id_to_token
                        print("\n✓ Verification passed: Loaded tokenizer matches original!")


