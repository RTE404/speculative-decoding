"""Load the target and draft models and check that they are compatible.

Facts checked here were confirmed from the models' Hugging Face files (plan, section 4):
both models have 151,936 output rows, of which only the first 151,665 are real tokens.
"""

from dataclasses import dataclass
from glob import glob
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

TARGET_ID = "Qwen/Qwen2.5-3B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

REAL_VOCAB = 151_665  # 151,643 regular tokens + 22 special tokens; rows above this are padding
EOS_IDS = (151_645, 151_643)  # <|im_end|>, <|endoftext|>


@dataclass
class ModelPair:
    target: PreTrainedModel
    draft: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    # The T4 has no bfloat16 support, and CPUs have no fast float16.
    return torch.float16 if device.type == "cuda" else torch.float32


def find_kaggle_model(variation: str, root: str = "/kaggle/input") -> str | None:
    """Return the folder of an attached Kaggle Model whose path contains `variation`, if any."""
    for config in sorted(glob(f"{root}/**/config.json", recursive=True)):
        folder = Path(config).parent
        if variation in str(folder).lower():
            return str(folder)
    return None


def load_model(name_or_path: str, device: torch.device, dtype: torch.dtype) -> PreTrainedModel:
    # One device only: device_map="auto" would split the model across both T4s and distort timings.
    # SDPA, not "eager": FlashAttention does not run on a T4, and eager attention is where
    # Qwen2's documented float16 overflow happens.
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    )
    model.eval()
    return model


def check_compatible(target: PreTrainedModel, draft: PreTrainedModel,
                     tokenizer: PreTrainedTokenizerBase, draft_tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target and draft tokenizers differ")
    if len(tokenizer) != REAL_VOCAB:
        raise ValueError(f"expected {REAL_VOCAB} real tokens, tokenizer has {len(tokenizer)}")
    for name, model in (("target", target), ("draft", draft)):
        if model.config.vocab_size < REAL_VOCAB:
            raise ValueError(f"{name} has only {model.config.vocab_size} output rows")
    for eos in EOS_IDS:
        if tokenizer.convert_ids_to_tokens(eos) not in ("<|im_end|>", "<|endoftext|>"):
            raise ValueError(f"unexpected token at EOS id {eos}")


def load_pair(target_path: str = TARGET_ID, draft_path: str = DRAFT_ID,
              device: torch.device | None = None, dtype: torch.dtype | None = None,
              draft_device: torch.device | None = None) -> ModelPair:
    """Load both models. The draft goes on `device` too unless `draft_device` is given
    (the float32 exactness test needs both T4s, since the pair does not fit on one)."""
    device = device or default_device()
    dtype = dtype or default_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_path)
    target = load_model(target_path, device, dtype)
    draft = load_model(draft_path, draft_device or device, dtype)
    check_compatible(target, draft, tokenizer, draft_tokenizer)
    return ModelPair(target, draft, tokenizer, device, dtype)


def encode_prompt(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    """Apply the chat template (which adds Qwen's default system prompt) and return input ids."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def real_logits(logits: torch.Tensor, vocab_size: int = REAL_VOCAB) -> torch.Tensor:
    """Drop the padding rows and move to float32, where all probability maths happens."""
    out = logits[..., :vocab_size].float()
    if not torch.isfinite(out).all():
        raise FloatingPointError("model produced inf or NaN logits (float16 overflow?)")
    return out
