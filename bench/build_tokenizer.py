"""Build a HuggingFace tokenizer from a Strata pack's GGUF tokenizer files.

Strata's pack ships vocab.json + merges.txt + a small tokenizer.json *descriptor*
(no HF tokenizer_config.json).  vLLM's benchmark needs an AutoTokenizer, so we
reconstruct the Qwen35 byte-level BPE here.  The pre-tokenizer regex is copied
verbatim from the descriptor's `pre_pattern` field.

Usage:
  python3 build_tokenizer.py --src <pack>/tokenizer --out bench/tokenizer
"""
import argparse
import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
SRC = args.src
OUT = args.out
os.makedirs(OUT, exist_ok=True)

vocab = json.load(open(os.path.join(SRC, "vocab.json"), encoding="utf-8"))
merges = []
with open(os.path.join(SRC, "merges.txt"), encoding="utf-8") as f:
    first = f.readline().strip()
    if not first.startswith("#"):
        a, b = first.split()
        merges.append((a, b))
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        a, b = line.split()
        merges.append((a, b))

PATTERN = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None, fuse_unk=False))
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()
tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
tok.save(os.path.join(OUT, "tokenizer.json"))

cfg = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "model_max_length": 262144,
    "add_bos_token": False,
    "add_eos_token": False,
    "bos_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "clean_up_tokenization_spaces": False,
}
json.dump(cfg, open(os.path.join(OUT, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=1)
json.dump({"eos_token": "<|im_end|>", "bos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"},
          open(os.path.join(OUT, "special_tokens_map.json"), "w"), ensure_ascii=False, indent=1)
print("wrote", OUT, "vocab", len(vocab), "merges", len(merges))
