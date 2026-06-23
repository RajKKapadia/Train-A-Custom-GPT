import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import ClassVar, cast

import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from generate import load_model, resolve_checkpoint
from src.config import CONFIG
from src.create_model import GPT, LayerCaches, sample_next_token


class GenerateStreamRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=4_000)
    max_new_tokens: int = Field(default=CONFIG.test.max_new_tokens, ge=1, le=1_000)
    temperature: float = Field(default=CONFIG.test.temperature, ge=0.0, le=5.0)
    top_k: int | None = Field(default=CONFIG.test.top_k, ge=0, le=1_000)
    repetition_penalty: float = Field(
        default=CONFIG.test.repetition_penalty,
        gt=0.0,
        le=10.0,
    )
    no_repeat_ngram_size: int = Field(
        default=CONFIG.test.no_repeat_ngram_size,
        ge=0,
        le=20,
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class ModelState:
    model: GPT | None = None
    tokenizer: tiktoken.Encoding | None = None
    device: str | None = None
    checkpoint: dict[str, object] | None = None
    checkpoint_path: str | None = None
    lock: asyncio.Lock | None = None


state = ModelState()


def _select_device() -> str:
    if CONFIG.train.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _loaded_model() -> tuple[GPT, tiktoken.Encoding, str, asyncio.Lock]:
    if (
        state.model is None
        or state.tokenizer is None
        or state.device is None
        or state.lock is None
    ):
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return state.model, state.tokenizer, state.device, state.lock


def _stream_token_ids_cached(
    model: GPT,
    idx: torch.Tensor,
    request: GenerateStreamRequest,
    eos_token_id: int | None,
) -> Iterator[int]:
    past_key_values: LayerCaches | None = None
    next_input: torch.Tensor | None = None

    with torch.inference_mode():
        for _ in range(request.max_new_tokens):
            if past_key_values is None:
                model_input = (
                    idx
                    if idx.size(1) <= model.config.block_size
                    else idx[:, -model.config.block_size :]
                )
                logits, past_key_values = model.forward_with_cache(model_input)
            else:
                cache_length = past_key_values[0][0].size(2)
                if cache_length >= model.config.block_size:
                    model_input = idx[:, -model.config.block_size :]
                    logits, past_key_values = model.forward_with_cache(model_input)
                else:
                    if next_input is None:
                        raise RuntimeError("Cached generation is missing next input")
                    logits, past_key_values = model.forward_with_cache(
                        next_input,
                        past_key_values,
                    )

            idx_next = sample_next_token(
                logits=logits[:, -1, :],
                input_ids=idx,
                temperature=request.temperature,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            next_input = idx_next

            token_id = int(idx_next[0, 0].item())
            yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                break


async def _generate_events(request: GenerateStreamRequest) -> AsyncIterator[str]:
    model, tokenizer, device, lock = _loaded_model()
    prompt_ids = tokenizer.encode(request.prompt)

    if len(prompt_ids) == 0:
        yield _sse("error", {"detail": "Prompt produced no tokens"})
        return

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    async with lock:
        try:
            yield _sse(
                "start",
                {
                    "checkpoint": state.checkpoint_path,
                    "device": device,
                    "prompt_tokens": len(prompt_ids),
                },
            )

            generated_tokens = 0
            for token_id in _stream_token_ids_cached(
                model=model,
                idx=input_ids,
                request=request,
                eos_token_id=tokenizer.eot_token,
            ):
                if token_id == tokenizer.eot_token:
                    break

                text = tokenizer.decode([token_id])
                generated_tokens += 1
                yield _sse(
                    "token",
                    {
                        "token": text,
                        "token_id": token_id,
                        "generated_tokens": generated_tokens,
                    },
                )
                await asyncio.sleep(0)

            yield _sse("done", {"generated_tokens": generated_tokens})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    device = _select_device()
    checkpoint_path = resolve_checkpoint(None)
    tokenizer = tiktoken.get_encoding(CONFIG.dataset.tokenizer_name)
    model, checkpoint = cast(
        tuple[GPT, dict[str, object]],
        load_model(checkpoint_path, device),
    )

    state.model = model
    state.tokenizer = tokenizer
    state.device = device
    state.checkpoint = checkpoint
    state.checkpoint_path = str(checkpoint_path)
    state.lock = asyncio.Lock()

    yield


app = FastAPI(title="TinyStories GPT API", lifespan=lifespan)


@app.post("/generate/stream")
async def generate_stream(request: GenerateStreamRequest) -> StreamingResponse:
    _ = _loaded_model()
    return StreamingResponse(
        _generate_events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
