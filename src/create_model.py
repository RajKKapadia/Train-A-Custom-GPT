"""Create a small decoder-only GPT model from config."""

import math
from typing import Any, TypeAlias, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]
LayerCaches: TypeAlias = tuple[KVCache, ...]


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits

    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    min_value = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_value, torch.full_like(logits, float("-inf")), logits
    )


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0")
    if repetition_penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for batch_idx, batch_tokens in enumerate(input_ids.tolist()):
        token_ids = sorted(set(batch_tokens))
        if not token_ids:
            continue

        token_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=adjusted.device,
        )
        token_logits = adjusted[batch_idx, token_tensor]
        adjusted[batch_idx, token_tensor] = torch.where(
            token_logits < 0,
            token_logits * repetition_penalty,
            token_logits / repetition_penalty,
        )

    return adjusted


def _banned_ngram_tokens(
    input_ids: torch.Tensor,
    no_repeat_ngram_size: int,
) -> list[list[int]]:
    if no_repeat_ngram_size <= 0:
        return [[] for _ in range(input_ids.size(0))]

    prefix_size = no_repeat_ngram_size - 1
    banned_tokens: list[list[int]] = []

    for batch_tokens in input_ids.tolist():
        generated_ngrams: dict[tuple[int, ...], list[int]] = {}

        for start in range(len(batch_tokens) - no_repeat_ngram_size + 1):
            ngram = tuple(batch_tokens[start : start + no_repeat_ngram_size])
            prefix = ngram[:-1]
            generated_ngrams.setdefault(prefix, []).append(ngram[-1])

        current_prefix = (
            tuple(batch_tokens[-prefix_size:]) if prefix_size > 0 else tuple()
        )
        banned_tokens.append(generated_ngrams.get(current_prefix, []))

    return banned_tokens


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size cannot be negative")
    if no_repeat_ngram_size == 0:
        return logits

    adjusted = logits.clone()
    for batch_idx, banned_tokens in enumerate(
        _banned_ngram_tokens(input_ids, no_repeat_ngram_size)
    ):
        if not banned_tokens:
            continue
        adjusted[
            batch_idx,
            torch.tensor(banned_tokens, dtype=torch.long, device=adjusted.device),
        ] = -float("inf")

    return adjusted


def sample_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
    logits = apply_no_repeat_ngram(logits, input_ids, no_repeat_ngram_size)

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
        self.bias: torch.Tensor

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        total_length = k.size(2)
        if total_length > self.bias.size(-1):
            raise ValueError(
                f"KV cache length {total_length} exceeds block_size {self.bias.size(-1)}"
            )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(
            self.bias[:, :, past_length : past_length + T, :total_length] == 0,
            float("-inf"),
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, present_key_value = self.attn.forward_with_cache(
            self.ln_1(x),
            past_key_value,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_key_values: LayerCaches | None = None,
    ) -> tuple[torch.Tensor, LayerCaches]:
        _, T = idx.size()
        past_length = 0

        if past_key_values is not None:
            if len(past_key_values) != len(self.transformer.h):
                raise ValueError(
                    f"Expected {len(self.transformer.h)} cache layers, "
                    f"got {len(past_key_values)}"
                )
            past_length = past_key_values[0][0].size(2)

        total_length = past_length + T
        if total_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {total_length} exceeds block_size "
                f"{self.config.block_size}"
            )

        pos = torch.arange(
            past_length,
            total_length,
            dtype=torch.long,
            device=idx.device,
        )
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        present_key_values: list[KVCache] = []
        for layer_idx, raw_block in enumerate(self.transformer.h):
            block = cast(Block, raw_block)
            layer_past = (
                None if past_key_values is None else past_key_values[layer_idx]
            )
            x, present_key_value = block.forward_with_cache(x, layer_past)
            present_key_values.append(present_key_value)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, tuple(present_key_values)

    def forward(self, idx, targets: torch.Tensor | None = None):
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def _generate_uncached(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 4,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            idx_next = sample_next_token(
                logits=logits[:, -1, :],
                input_ids=idx,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and bool(torch.all(idx_next == eos_token_id)):
                break
        return idx

    def _generate_cached(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 4,
    ) -> torch.Tensor:
        past_key_values: LayerCaches | None = None
        next_input: torch.Tensor | None = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_input = (
                    idx
                    if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size :]
                )
                logits, past_key_values = self.forward_with_cache(model_input)
            else:
                cache_length = past_key_values[0][0].size(2)
                if cache_length >= self.config.block_size:
                    model_input = idx[:, -self.config.block_size :]
                    logits, past_key_values = self.forward_with_cache(model_input)
                else:
                    if next_input is None:
                        raise RuntimeError("Cached generation is missing next input")
                    logits, past_key_values = self.forward_with_cache(
                        next_input,
                        past_key_values,
                    )

            idx_next = sample_next_token(
                logits=logits[:, -1, :],
                input_ids=idx,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            next_input = idx_next

            if eos_token_id is not None and bool(torch.all(idx_next == eos_token_id)):
                break

        return idx

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 4,
        use_cache: bool = True,
    ):
        if use_cache:
            return self._generate_cached(
                idx=idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        return self._generate_uncached(
            idx=idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )


def create_model(config: ModelConfig) -> GPT:
    return GPT(config)


def model_config_from_dict(d: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**d)
