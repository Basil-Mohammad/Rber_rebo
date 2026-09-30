"""A tiny random Qwen2 model + word-level tokenizer for offline tests of the LM planner (no downloads)."""
import json, os, re, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from rber.domain import prompt_text


def build(world):
    words = set()
    for t in world.train + world.held:
        words |= set(re.findall(r"\w+|[^\w\s]", prompt_text(world, t)))
    for k in range(1, world.K + 1):
        words |= {str(k)}
    words |= {"Step", ":", "user", "assistant"}
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {"<unk>": 0}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    for s in special:
        vocab[s] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(special)
    ftok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|endoftext|>", eos_token="<|im_end|>", unk_token="<unk>")
    ftok.chat_template = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024, eos_token_id=vocab["<|im_end|>"],
                      pad_token_id=vocab["<|endoftext|>"], tie_word_embeddings=True)
    d = tempfile.mkdtemp(); Qwen2ForCausalLM(cfg).save_pretrained(d); ftok.save_pretrained(d)
    p = os.path.join(d, "tokenizer_config.json"); c = json.load(open(p)); c["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(c, open(p, "w"))
    return d
