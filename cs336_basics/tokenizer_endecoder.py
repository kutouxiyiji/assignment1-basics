# The encoder and decoder for given tokenizer (e.g. BPE tokenizer trained from tokenizer.py)
from typing import Iterator, Iterable
import json
from pathlib import Path
import regex as re
# Import PAT from tokenizer.py in the same directory
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizer import PAT


def gpt2_bytes_to_unicode():
        """
        Returns a mapping between every possible byte (an integer from 0 to 255) to a
        printable unicode string character representation.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
                if b not in bs:
                        bs.append(b)
                        cs.append(2**8 + n)
                        n += 1
        characters = [chr(n) for n in cs]
        d = dict(zip(bs, characters))
        return d


class TokenizerEnDeCoder():
        def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None) -> None:
                self.id_to_token = vocab.copy()
                self.token_to_id = {vocab[id]: id for id in vocab}
                self.merges = merges
                self.merges_set = set(merges)
                self.special_tokens = special_tokens
                
                # Add special tokens to vocabulary if they don't exist
                if self.special_tokens:
                        for special_token in self.special_tokens:
                                byte_encoded_special_token = special_token.encode("utf-8")
                                if byte_encoded_special_token not in self.token_to_id:
                                        # Add to vocabulary
                                        new_id = len(self.id_to_token)
                                        self.id_to_token[new_id] = byte_encoded_special_token
                                        self.token_to_id[byte_encoded_special_token] = new_id
                        
                        # Sort special tokens by length (longest first) to handle overlapping tokens
                        # for example, <s_token><s_token> will be split before the other shorter speical token <s_token>
                        # see tests/test_tokenizer.py::test_overlapping_special_tokens
                        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
                        escaped_tokens = [re.escape(special_token) for special_token in sorted_tokens]
                        self.escaped_pattern = "|".join(escaped_tokens)
                else:
                        self.escaped_pattern = ""
                self.split_special_token =  b'<|endoftext|>'
                self.chunk_size = 4096


        @classmethod
        def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
                # Load vocab from JSON
                vocab_path = Path(vocab_filepath)
                with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocab_json = json.load(f)
                
                # Get GPT-2 unicode to byte mapping (reverse of byte_encoder)
                byte_encoder = gpt2_bytes_to_unicode()
                byte_decoder = {v: k for k, v in byte_encoder.items()}
                
                # Helper function to convert GPT-2 string to bytes
                def str_to_bytes(token_str):
                        """Convert GPT-2 encoded string back to bytes"""
                        return bytes([byte_decoder[c] for c in token_str])
                
                # Convert vocab from {str_token: int_id} to {int_id: bytes_token}
                id_to_token = {}
                for token_str, token_id in vocab_json.items():
                        # Convert string token to bytes using GPT-2 decoding
                        token_bytes = str_to_bytes(token_str)
                        id_to_token[token_id] = token_bytes
                
                # Load merges from txt file
                merges_path = Path(merges_filepath)
                merges_list = []
                with open(merges_path, 'r', encoding='utf-8') as f:
                        for line in f:
                                line = line.strip()
                                # Skip empty lines and comments
                                if not line or line.startswith('#'):
                                        continue
                                
                                # Parse merge: "token1 token2"
                                parts = line.split()
                                if len(parts) == 2:
                                        token1 = str_to_bytes(parts[0])
                                        token2 = str_to_bytes(parts[1])
                                        merges_list.append((token1, token2))
                
                # Create and return tokenizer instance
                return cls(id_to_token, merges_list, special_tokens)

        def _remove_special_tokens(self, input_text):
                if not self.special_tokens:
                        return input_text
                return re.sub(self.escaped_pattern, "", input_text)
        
        def _split_with_special_tokens(self, text: str) -> list[str]:
                if not self.special_tokens:
                        return [text]
                pattern = f"({self.escaped_pattern})" # with (), it will perform as capturing group, and the spliter will be included in the retsults
                # Split by pattern, keeping the special tokens
                parts = re.split(pattern, text)
                # Filter out empty strings
                result = [part for part in parts if part]
                return result
        
        # main merge logic.
        def _merge_init_token_bytes(self, init_token_bytes):
                if len(init_token_bytes) <= 1:
                        return init_token_bytes
                # scan merges, if one match, merge it. Then scan again
                changed = True
                while changed:
                        changed = False
                        for merge in self.merges:
                                i = 0
                                new_bytes = []
                                while i < len(init_token_bytes):
                                        if i < len(init_token_bytes) - 1 and init_token_bytes[i] == merge[0] and init_token_bytes[i+1] == merge[1]:
                                                new_bytes.append(merge[0] + merge[1])
                                                i += 2
                                                changed = True
                                        else:
                                                new_bytes.append(init_token_bytes[i])
                                                i += 1
                                init_token_bytes = new_bytes
                                if changed:
                                        break
                return init_token_bytes

        def _process_and_merge_sub_text(self, sub_text):
                # split by PAT
                init_token_bytes = []
                for match in re.finditer(PAT, sub_text):
                        pre_token = match.group()
                        # Convert to bytes using UTF-8 encoding
                        token_bytes = pre_token.encode("UTF-8")
                        # Convert each byte to individual tokens
                        for byte_val in token_bytes:
                                init_token_bytes.append(bytes([byte_val]))
                
                if len(init_token_bytes) <= 1:
                        # Convert single byte to token ID
                        if init_token_bytes:
                                return [self.token_to_id[init_token_bytes[0]]]
                        return []
                
                # merge init_token_bytes
                merged_token_bytes = self._merge_init_token_bytes(init_token_bytes)
                merged_token_ids = []
                for token_byte in merged_token_bytes:
                        merged_token_ids.append(self.token_to_id[token_byte])
                return merged_token_ids


        def _process_chunk_text(self, chunk_text):
                # split by special tokens, [sub_text, special_token, sub_text, ...]
                # process each sub text and become the pre_token saperated by PAT [pre_tokens, special_token, pre_tokens, ...]
                # pre_tokens becom init_token_ids, perform encoder (i.e. merge) later in encode_one_chunk.
                # in encode_one_chunk, it became [merged_tokens, speical_token, merged_tokens, special_token...]
                # flat and return
                sub_texts = self._split_with_special_tokens(chunk_text)
                # print(f"sub texts are {sub_texts}")
                token_ids_list = [] # final bytes list
                for sub_text in sub_texts:
                        if self.special_tokens and sub_text in self.special_tokens:
                                # Look up special token by its byte representation
                                special_token_bytes = sub_text.encode("utf-8")
                                special_token_id = self.token_to_id[special_token_bytes]
                                token_ids_list.append(special_token_id)
                        else:
                                merged_token_ids = self._process_and_merge_sub_text(sub_text)
                                token_ids_list += merged_token_ids
                return token_ids_list
                # return self._process_and_merge_sub_text(sub_text)

        # we just need to _process_chunk_text, for the chunk text from the input, then it's encoded
        # we will chunk the input text first


        def encode(self, text: str) -> list[int]:
                return self._process_chunk_text(text)

        def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
                chunk = []
                split_token_str = self.split_special_token.decode('UTF-8')
                for item in iterable:
                        chunk.append(item)
                        if (len(chunk) >= self.chunk_size and item == split_token_str) or (len(chunk) >= 5* self.chunk_size):
                                chunk_text = "".join(chunk)
                                chunk_tokens = self._process_chunk_text(chunk_text)
                                yield from chunk_tokens
                                chunk = []
                if chunk:
                        chunk_text = "".join(chunk)
                        chunk_tokens = self._process_chunk_text(chunk_text)
                        yield from chunk_tokens
        
        def decode(self, ids: list[int]) -> str:
                bytes_list = []
                for id in ids:
                        bytes_list.append(self.id_to_token.get(id, '\uFFFD'.encode('utf-8')))
                joint_bytes = b''.join(bytes_list)
                return joint_bytes.decode('utf-8', errors='replace')


# endecoder = TokenizerEnDeCoder.from_files('./tests/fixtures/gpt2_vocab.json', './tests/fixtures/gpt2_merges.txt', ['<|endoftext|>'])
# print()
# # print("test the encoder. \nLet's play dota2.^86774$%345354%^&&")
# print(endecoder.encode("s"))
# print(endecoder.decode([82]))
# # print(endecoder.id_to_token)

# texts = ["Hello world!", "This is a test.", "Another sentence."]
# combined_text = "".join(texts)
# print()
# print("the test for the iter encoder")
# print(list(endecoder.encode_iterable(texts)))
# print("the test for the combined text")
# encoded = endecoder.encode(combined_text)
# print(encoded)
# print(f"the compression rate is {len(combined_text.encode("UTF-8")) / len(encoded)}")
# print()
# print("start to decode.")
# print(endecoder.decode(encoded + [2000]))
# for match in re.finditer(PAT, "Letplay dota2."):
#                 pre_token = match.group()
#                 init_token_ids = tuple(pre_token.encode("UTF-8"))
#                 print(PAT)
#                 print(pre_token)
#                 print(init_token_ids)