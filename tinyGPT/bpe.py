import os
import json
import regex as re
import requests

import torch


def bytes_to_unicode() -> dict[int,str]:
    """
    Maps all bytes to unicode characters that can be rendered nicely

    Any bytes that are not rendered nicely are mapped to bytes past 256 (Ā onwards)

    Returns:
        dictionary mapping bytes to renderable unicode characters
    """
    nice_bytes = list(range(ord("!"), ord("~")+1)) + \
            list(range(ord("¡"), ord("¬")+1)) + \
            list(range(ord("®"), ord("ÿ")+1))
    all_bytes = nice_bytes[:]

    byte_idx = 0
    for byte in range(2**8):
        if byte not in nice_bytes:
            nice_bytes.append(byte)
            all_bytes.append(2**8 + byte_idx)  # convert ugly byte to next available byte
            byte_idx += 1
    
    all_bytes_chars = [chr(n) for n in all_bytes]  # map all bytes to unicode
    byte_to_char = dict(zip(nice_bytes, all_bytes_chars))

    return byte_to_char


def get_pairs(word: tuple) -> set[tuple]:
    """
    Get all bigrams for given word

    Args:
        word: word to compute bigrams from
    
    Returns:
        set of tuples containing the characters for each bigram
    """
    return set([(a, b) for a, b in zip(word, word[1:])])


class Encoder:
    def __init__(self, encoder: dict, bpe_merges: list[tuple]):
        """
        Args:
            encoder: dictionary mqapping tokens to token indicies
            bpe_merges: list of bi-gram tokens to be merged for BPE
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}

        # map token merges to an index
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        """
        - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
        - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces
        """
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}  # cache mapping a token to a bpe token


    def bpe(self, token: str) -> str:
        """
        this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens
        up the tree. token is a string of one individual 'word' (after regex tokenization)
        and after byte encoding, e.g. 'Ġthere'.

        Args:
            token: a string of one individual 'word', after byte encoding, e.g. 'Ġthere'

        Returns:
            string containing the byte-encoded tokens, seperated by a space
        """
        if token in self.cache: 
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # find the next lowest rank bigram that can be merged
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # no more bigrams are eligible to be merged
            first, second = bigram

            # we will now replace all occurences of the bigram into a merged token
            new_word = []
            i = 0
            while i < len(word):
                # find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        
        return word


    def encode(self, text) -> list[tuple[str,str]]:
        """
        Encode text into a list of integers

        Args:
            text: string to encode

        Returns:
            list of token indexes for each BPE token
        """
        bpe_idx = []
        tokens = re.findall(self.pat, text)  # pre-tokenize the input text into string tokens (words, roughly speaking)
        
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
        
        return bpe_idx


    def encode_and_show_work(self, text):
        """debugging function, same as encode but returns all intermediate work """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        return {
            'bpe_idx': bpe_idx, # the actual output sequence
            'tokens': tokens, # result of pre-tokenization
            'parts': parts, # intermediates for each token part
        }


    def decode(self, bpe_idx):
        """
        Decode token indicies to unicode tokens
        
        Args:
            bpe_idx: BPE index of token to decode
        
        Returns:
            decoded text from BPE index
        """
        # inverse map the integers to get the tokens
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode('utf-8', errors='replace')
        
        return text


class BPETokenizer:
    """ PyTorch-aware class that wraps the Encoder above """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        assert return_tensors == 'pt'
        assert isinstance(text, str)
        idx = [self.encoder.encode(text)]
        out = torch.tensor(idx, dtype=torch.long)
        
        return out

    def decode(self, idx):
        assert idx.ndim == 1
        text = self.encoder.decode(idx.tolist())
        
        return text


def download_file(local_file: str, remote_file: str) -> None:
    """
    Downloads remote_file to local_file if necessary
    
    Args:
        local_file: local path to save file to
        remote_file: remote file to download from
    """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)


def get_encoder() -> Encoder:
    """
    Fetches an instance of the GPT BPE encoder/decoder from openai

    Returns:
        an instance of the encoder
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'tinygpt')
    os.makedirs(cache_dir, exist_ok=True)

    # load encoder.json that has the raw mappings from token -> bpe index
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    download_file(encoder_local_file, encoder_remote_file)
    
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257 # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token

    # load vocab.bpe that contains the bpe merge tree
    # each line are two tokens that are to be merged (as found by the BPE algorithm)
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    download_file(vocab_local_file, vocab_remote_file)
    
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read().split('\n')[1:-1]  # strip the version on first line and blank last line
    
    # parse token pairs into list of tuples
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data]
    assert len(bpe_merges) == 50000 # 50,000 merged tokens

    return Encoder(encoder, bpe_merges)


if __name__ == '__main__':
    text = "Hello!! This is a test of the tokeniser. It's 2026. w00t :D 🤗"
    encoder = get_encoder()
    result = encoder.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("\nFirst the text gets pre-tokenized, broken up into chunks, the outcome is:")
    print(result['tokens'])
    
    print("\nThen we iterate over each chunk and process them in turn...")
    for part in result['parts']:
        print(part)
    
    print("\nAnd the final outcome is concatenating and flattening all the token_ix:")
    print(result['bpe_idx'])
    print("\nReady to feed into a Transformer!")