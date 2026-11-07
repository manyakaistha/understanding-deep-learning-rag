"""Handwritten local dense embedding and vector utilities for the UDL pipeline.

Uses direct PyTorch + HuggingFace Transformers with `Qwen/Qwen3-Embedding-0.6B`,
pinned revision, left-padded tokenization, last-valid-token pooling, L2
normalization, and float32 SQLite BLOB packing.

Long documents (> `max_length` tokens) are split into overlapping token windows
(`window_overlap=128`), encoded across all windows without dropping any tail
tokens, and aggregated into a single L2-normalized mean vector per document.
Queries exceeding `max_length` tokens are rejected explicitly.

Inference disables KV cache (`model.config.use_cache = False`), uses memory-safe
`model_dtype` defaults (`float16` on MPS/CUDA, `float32` on CPU, recorded in the
profile and cache identity), conservative token-budget subbatching, and
automatic microbatch split + CPU fallback on accelerator OOM without ever
truncating content.

No `sentence-transformers`, `scipy`, or vector-search libraries are used.
Model weights are loaded lazily only when document or query encoding is invoked.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import struct
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.contracts import DEFAULT_EMBEDDING_MODEL, UDLException

DEFAULT_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
DEFAULT_DOCUMENT_PROMPT = ""
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 2
DEFAULT_WINDOW_OVERLAP = 128
DEFAULT_OVERFLOW_POLICY = "sliding_window"
DEFAULT_WINDOW_AGGREGATION = "l2_normalized_mean"
DEFAULT_QUERY_OVERFLOW_POLICY = "reject"

_DTYPE_ALIASES: Dict[str, str] = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "torch.float32": "float32",
}


def _normalize_device_type(device: Optional[str]) -> str:
    if not device:
        return "cpu"
    clean = str(device).strip().lower()
    if clean.startswith("mps"):
        return "mps"
    if clean.startswith("cuda"):
        return "cuda"
    return "cpu"


def resolve_model_dtype(
    model_dtype: Optional[str] = None,
    *,
    device: Optional[str] = None,
) -> str:
    """Resolve canonical model forward precision (`float16`, `bfloat16`, `float32`).

    Defaults to `float16` on MPS/CUDA for memory-safe attention and `float32` on CPU.
    """
    if model_dtype is not None and str(model_dtype).strip().lower() not in {"", "auto"}:
        raw = str(model_dtype).strip().lower()
        if raw not in _DTYPE_ALIASES:
            raise ValueError(
                f"Unsupported model_dtype '{model_dtype}'. "
                "Expected one of: float16, bfloat16, float32, auto."
            )
        return _DTYPE_ALIASES[raw]
    dev_type = _normalize_device_type(device)
    if dev_type in {"mps", "cuda"}:
        return "float16"
    return "float32"


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "mps backend out of memory" in msg
        or "cuda out of memory" in msg
        or "invalid buffer size" in msg
    )


def _empty_device_cache(torch_module: Any, device: Optional[str] = None) -> None:
    if torch_module is None:
        return
    dev_type = _normalize_device_type(device)
    try:
        if dev_type == "mps" and hasattr(torch_module, "mps") and hasattr(torch_module.mps, "empty_cache"):
            torch_module.mps.empty_cache()
        elif dev_type == "cuda" and hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
            torch_module.cuda.empty_cache()
    except Exception:
        pass


def format_query_prompt(query: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """Format a search query with the exact Qwen3-Embedding instruction prefix."""
    clean_query = "" if query is None else str(query)
    clean_instruction = (instruction or DEFAULT_QUERY_INSTRUCTION).strip()
    if clean_instruction.startswith("Instruct:"):
        prefix = clean_instruction
        if not prefix.endswith("Query:"):
            prefix = f"{prefix}\nQuery:"
        return f"{prefix}{clean_query}"
    return f"Instruct: {clean_instruction}\nQuery:{clean_query}"


def format_document_prompt(
    document: str, document_prompt: str = DEFAULT_DOCUMENT_PROMPT
) -> str:
    """Format a document string for embedding (empty prefix for Qwen3-Embedding)."""
    clean_doc = "" if document is None else str(document)
    if not document_prompt:
        return clean_doc
    return f"{document_prompt}{clean_doc}"


def l2_normalize(vector: Sequence[float]) -> List[float]:
    """L2-normalize a numeric vector in pure Python and validate finiteness."""
    if not vector:
        raise UDLException(
            "embedding_empty_vector",
            "Embedding vector cannot be empty.",
            retryable=False,
        )
    vals = [float(v) for v in vector]
    if not all(math.isfinite(v) for v in vals):
        raise UDLException(
            "embedding_nonfinite_vector",
            "Embedding vector contains NaN or infinite values.",
            retryable=False,
        )
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 0.0 or not math.isfinite(norm):
        raise UDLException(
            "embedding_zero_norm",
            "Embedding vector has zero L2 norm.",
            retryable=False,
        )
    return [v / norm for v in vals]


def pack_vector(
    vector: Sequence[float], *, expected_dim: Optional[int] = None
) -> bytes:
    """Validate and pack a float vector into a little-endian float32 SQLite BLOB."""
    if not vector:
        raise UDLException(
            "embedding_empty_vector",
            "Cannot pack an empty embedding vector.",
            retryable=False,
        )
    if expected_dim is not None and len(vector) != int(expected_dim):
        raise UDLException(
            "embedding_dimension_mismatch",
            f"Expected embedding dimension {expected_dim}, got {len(vector)}.",
            retryable=False,
        )
    vals = [float(v) for v in vector]
    if not all(math.isfinite(v) for v in vals):
        raise UDLException(
            "embedding_nonfinite_vector",
            "Cannot pack non-finite values into float32 BLOB.",
            retryable=False,
        )
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 0.0:
        raise UDLException(
            "embedding_zero_norm",
            "Cannot pack a zero-norm embedding vector.",
            retryable=False,
        )
    return struct.pack(f"<{len(vals)}f", *vals)


def unpack_vector(
    blob: Union[bytes, bytearray, memoryview], *, expected_dim: Optional[int] = None
) -> Tuple[float, ...]:
    """Unpack a little-endian float32 SQLite BLOB into a tuple of floats."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise UDLException(
            "embedding_corrupt_blob",
            "Embedding BLOB must be bytes-like.",
            retryable=False,
        )
    byte_len = len(blob)
    if byte_len == 0 or byte_len % 4 != 0:
        raise UDLException(
            "embedding_corrupt_blob",
            f"Invalid float32 BLOB byte length: {byte_len}.",
            retryable=False,
        )
    dim = byte_len // 4
    if expected_dim is not None and dim != int(expected_dim):
        raise UDLException(
            "embedding_dimension_mismatch",
            f"Expected unpacked embedding dimension {expected_dim}, got {dim}.",
            retryable=False,
        )
    values = struct.unpack(f"<{dim}f", blob)
    if not all(math.isfinite(v) for v in values):
        raise UDLException(
            "embedding_nonfinite_vector",
            "Unpacked embedding BLOB contains non-finite values.",
            retryable=False,
        )
    return values


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute exact cosine similarity between two equal-dimension vectors."""
    if len(vec_a) != len(vec_b):
        raise UDLException(
            "embedding_dimension_mismatch",
            f"Cosine similarity dimension mismatch: {len(vec_a)} vs {len(vec_b)}.",
            retryable=False,
        )
    if not vec_a:
        raise UDLException(
            "embedding_empty_vector",
            "Cannot compute cosine similarity on empty vectors.",
            retryable=False,
        )
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for a, b in zip(vec_a, vec_b):
        fa = float(a)
        fb = float(b)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            raise UDLException(
                "embedding_nonfinite_vector",
                "Cannot compute cosine similarity with non-finite values.",
                retryable=False,
            )
        dot += fa * fb
        norm_a_sq += fa * fa
        norm_b_sq += fb * fb
    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        raise UDLException(
            "embedding_zero_norm",
            "Cannot compute cosine similarity with a zero-norm vector.",
            retryable=False,
        )
    return dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))


def last_valid_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Pool the last valid (non-padding) token from each sequence.

    Supports both left-padded and right-padded batches in PyTorch, with a pure
    Python fallback for lightweight unit tests when tensors are sequences.
    """
    if hasattr(last_hidden_states, "shape") and hasattr(attention_mask, "shape"):
        import torch

        if last_hidden_states.ndim != 3 or attention_mask.ndim != 2:
            raise ValueError(
                "last_hidden_states must be 3D (B, S, D) and attention_mask 2D (B, S)"
            )
        if (
            last_hidden_states.shape[0] != attention_mask.shape[0]
            or last_hidden_states.shape[1] != attention_mask.shape[1]
        ):
            raise ValueError("Batch and sequence dimensions must match")
        valid_counts = attention_mask.sum(dim=1)
        if bool((valid_counts <= 0).any()):
            raise ValueError("Each sequence in attention_mask must have >= 1 valid token")
        left_padding = bool(
            (attention_mask[:, -1].sum() == attention_mask.shape[0])
        )
        if left_padding:
            return last_hidden_states[:, -1]
        seq_len = attention_mask.shape[1]
        positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
        valid_mask = (attention_mask > 0).long()
        last_indices = (positions * valid_mask).amax(dim=1)
        batch_indices = torch.arange(
            last_hidden_states.shape[0], device=last_hidden_states.device
        )
        return last_hidden_states[batch_indices, last_indices]

    # Pure-Python sequence fallback for unit tests without torch tensors.
    pooled = []
    for row_states, row_mask in zip(last_hidden_states, attention_mask):
        last_idx = -1
        for idx, mask_val in enumerate(row_mask):
            if int(mask_val) > 0:
                last_idx = idx
        if last_idx < 0:
            raise ValueError("Sequence has no valid tokens in attention_mask")
        pooled.append(list(row_states[last_idx]))
    return pooled


def split_token_windows(
    token_ids: Sequence[int],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    window_overlap: int = DEFAULT_WINDOW_OVERLAP,
) -> List[List[int]]:
    """Split a token ID sequence into windows of length <= `max_length` with `window_overlap`.

    Every input token appears in at least one window; no tail tokens are dropped.
    """
    max_len = int(max_length)
    overlap = int(window_overlap)
    if max_len <= 0:
        raise ValueError("max_length must be positive")
    if overlap < 0 or overlap >= max_len:
        raise ValueError("window_overlap must satisfy 0 <= window_overlap < max_length")

    ids = [int(t) for t in token_ids]
    if len(ids) <= max_len:
        return [ids]

    stride = max_len - overlap
    windows: List[List[int]] = []
    start = 0
    total = len(ids)
    while start < total:
        end = min(start + max_len, total)
        windows.append(ids[start:end])
        if end >= total:
            break
        start += stride
    return windows


def aggregate_window_vectors(
    window_vectors: Sequence[Sequence[float]],
    *,
    aggregation: str = DEFAULT_WINDOW_AGGREGATION,
) -> List[float]:
    """Aggregate per-window L2-normalized vectors into a single L2-normalized document vector."""
    if not window_vectors:
        raise UDLException(
            "embedding_empty_vector",
            "Cannot aggregate an empty sequence of window vectors.",
            retryable=False,
        )
    if aggregation not in {"l2_normalized_mean", "mean"}:
        raise ValueError(f"Unsupported window aggregation mode: {aggregation}")

    normed_windows = [l2_normalize(vec) for vec in window_vectors]
    if len(normed_windows) == 1:
        return normed_windows[0]

    dim = len(normed_windows[0])
    count = float(len(normed_windows))
    mean_vec = [0.0] * dim
    for w_vec in normed_windows:
        if len(w_vec) != dim:
            raise UDLException(
                "embedding_dimension_mismatch",
                f"Window vector dimension mismatch: {len(w_vec)} vs {dim}.",
                retryable=False,
            )
        for idx, val in enumerate(w_vec):
            mean_vec[idx] += val / count
    return l2_normalize(mean_vec)


def canonical_profile_dict(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a canonical JSON-compatible dictionary for model configuration hashing.

    Preserves all standard fields including model precision (`model_dtype`) and
    long-document windowing/aggregation identity fields, plus any extra identity
    fields supplied by custom profiles.
    """
    query_inst = str(profile.get("query_instruction", DEFAULT_QUERY_INSTRUCTION))
    out_dtype = str(profile.get("dtype", "float32"))
    model_dtype = str(profile.get("model_dtype") or out_dtype)
    canonical: Dict[str, Any] = {
        "backend": str(profile.get("backend", "torch+transformers")),
        "model_name": str(profile.get("model_name", DEFAULT_EMBEDDING_MODEL)),
        "revision": str(
            profile.get("configured_revision")
            or profile.get("revision")
            or DEFAULT_EMBEDDING_REVISION
        ),
        "dimension": int(profile.get("dimension", DEFAULT_EMBEDDING_DIMENSION)),
        "pooling": str(profile.get("pooling", "last_valid_token")),
        "normalize": str(profile.get("normalize", "l2")),
        "dtype": out_dtype,
        "model_dtype": model_dtype,
        "max_length": int(profile.get("max_length", DEFAULT_MAX_LENGTH)),
        "overflow_policy": str(
            profile.get("overflow_policy", DEFAULT_OVERFLOW_POLICY)
        ),
        "window_overlap": int(
            profile.get("window_overlap", DEFAULT_WINDOW_OVERLAP)
        ),
        "window_aggregation": str(
            profile.get("window_aggregation", DEFAULT_WINDOW_AGGREGATION)
        ),
        "query_overflow_policy": str(
            profile.get("query_overflow_policy", DEFAULT_QUERY_OVERFLOW_POLICY)
        ),
        "query_instruction": query_inst,
        "query_prompt": str(
            profile.get(
                "query_prompt",
                format_query_prompt("", query_inst),
            )
        ),
        "document_prompt": str(
            profile.get("document_prompt", DEFAULT_DOCUMENT_PROMPT)
        ),
    }
    # Preserve any extra identity fields on custom profiles without dropping them
    for key, val in profile.items():
        if key == "configured_revision" or key in canonical:
            continue
        canonical[str(key)] = val
    return canonical


def compute_profile_fingerprint(profile: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint for an embedder profile."""
    canonical = canonical_profile_dict(profile)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embedding_cache_key(
    text: str, profile: Mapping[str, Any], *, is_query: bool = False
) -> str:
    """Compute a cache key from the full embedding input text and model config."""
    canonical_prof = canonical_profile_dict(profile)
    formatted_input = (
        format_query_prompt(text, canonical_prof["query_instruction"])
        if is_query
        else format_document_prompt(text, canonical_prof["document_prompt"])
    )
    payload = json.dumps(
        {
            "is_query": bool(is_query),
            "raw_text": "" if text is None else str(text),
            "formatted_input": formatted_input,
            "model_config": canonical_prof,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LocalEmbedder:
    """Lazy-loaded local embedder using direct PyTorch + Transformers.

    Defaults to `Qwen/Qwen3-Embedding-0.6B` at its pinned revision, using left
    padding, last-valid-token pooling, L2 normalization in float32, disabled KV
    cache (`model.config.use_cache = False`), memory-safe `model_dtype`
    (`float16` on MPS/CUDA, `float32` on CPU), and lossless sliding-window
    aggregation (`window_overlap=128`) for documents > `max_length`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        revision: str = DEFAULT_EMBEDDING_REVISION,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        document_prompt: str = DEFAULT_DOCUMENT_PROMPT,
        max_length: int = DEFAULT_MAX_LENGTH,
        window_overlap: int = DEFAULT_WINDOW_OVERLAP,
        overflow_policy: str = DEFAULT_OVERFLOW_POLICY,
        window_aggregation: str = DEFAULT_WINDOW_AGGREGATION,
        query_overflow_policy: str = DEFAULT_QUERY_OVERFLOW_POLICY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        model_dtype: Optional[str] = None,
        allow_cpu_fallback: bool = True,
        local_files_only: bool = False,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        if not model_name or not str(model_name).strip():
            raise ValueError("model_name must be a non-empty string")
        if not revision or not str(revision).strip():
            raise ValueError("revision must be a non-empty pinned revision string")
        if int(dimension) <= 0:
            raise ValueError("dimension must be positive")
        if int(max_length) <= 0:
            raise ValueError("max_length must be positive")
        if int(window_overlap) < 0 or int(window_overlap) >= int(max_length):
            raise ValueError("window_overlap must satisfy 0 <= window_overlap < max_length")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")

        self.model_name = str(model_name).strip()
        self.revision = str(revision).strip()
        self._resolved_revision = self.revision
        self.dimension = int(dimension)
        self.query_instruction = str(query_instruction).strip()
        self.document_prompt = str(document_prompt)
        self.max_length = int(max_length)
        self.window_overlap = int(window_overlap)
        self.overflow_policy = str(overflow_policy).strip()
        self.window_aggregation = str(window_aggregation).strip()
        self.query_overflow_policy = str(query_overflow_policy).strip()
        self.batch_size = int(batch_size)
        self.device = device
        self._configured_model_dtype = model_dtype
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.local_files_only = bool(local_files_only)

        self._tokenizer = tokenizer
        self._model = model
        self._inference_lock = threading.RLock()
        self._resolved_device: Optional[str] = device
        self._subbatch_count = 0
        self.cpu_fallback_triggered = False

        target_dev = self._detect_default_device()
        self.model_dtype = resolve_model_dtype(model_dtype, device=target_dev)

        if self._model is not None:
            cfg = getattr(self._model, "config", None)
            if cfg is not None:
                try:
                    cfg.use_cache = False
                except Exception:
                    pass
                commit_hash = getattr(cfg, "_commit_hash", None)
                if commit_hash:
                    self._resolved_revision = str(commit_hash)
                hidden_size = getattr(cfg, "hidden_size", None)
                if hidden_size is not None and int(hidden_size) != self.dimension:
                    raise UDLException(
                        "embedding_dimension_mismatch",
                        f"Model hidden_size {hidden_size} != configured dimension {self.dimension}.",
                        retryable=False,
                    )

    def _detect_default_device(self) -> str:
        if self.device:
            return str(self.device).strip()
        try:
            import torch

            return self._select_device(torch)
        except Exception:
            return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Return True if the tokenizer and model have been loaded into memory."""
        return self._tokenizer is not None and self._model is not None

    def format_query(self, query: str) -> str:
        """Apply the configured query instruction prefix."""
        return format_query_prompt(query, self.query_instruction)

    def format_document(self, text: str) -> str:
        """Apply the configured document prompt prefix."""
        return format_document_prompt(text, self.document_prompt)

    @property
    def profile(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary describing the embedder config."""
        return {
            "backend": "torch+transformers",
            "model_name": self.model_name,
            "revision": self._resolved_revision,
            "configured_revision": self.revision,
            "dimension": self.dimension,
            "pooling": "last_valid_token",
            "normalize": "l2",
            "dtype": "float32",
            "model_dtype": self.model_dtype,
            "max_length": self.max_length,
            "overflow_policy": self.overflow_policy,
            "window_overlap": self.window_overlap,
            "window_aggregation": self.window_aggregation,
            "query_overflow_policy": self.query_overflow_policy,
            "query_instruction": self.query_instruction,
            "query_prompt": format_query_prompt("", self.query_instruction),
            "document_prompt": self.document_prompt,
        }

    @property
    def fingerprint(self) -> str:
        """Return the deterministic SHA-256 fingerprint of this embedder's profile."""
        return compute_profile_fingerprint(self.profile)

    def cache_key_for(self, text: str, *, is_query: bool = False) -> str:
        """Return the full input + model config SHA-256 cache key."""
        return embedding_cache_key(text, self.profile, is_query=is_query)

    def _select_device(self, torch_module: Any) -> str:
        if self.device:
            return str(self.device).strip()
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module, "backends")
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    def _resolve_torch_dtype(self, torch_module: Any, dtype_name: str) -> Any:
        mapping = {
            "float16": getattr(torch_module, "float16", None),
            "bfloat16": getattr(torch_module, "bfloat16", None),
            "float32": getattr(torch_module, "float32", None),
        }
        resolved = mapping.get(dtype_name)
        if resolved is None:
            return torch_module.float32
        return resolved

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            if getattr(self._model, "config", None) is not None:
                try:
                    self._model.config.use_cache = False
                except Exception:
                    pass
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise UDLException(
                "embedding_runtime_missing",
                "LocalEmbedder requires PyTorch and HuggingFace Transformers "
                f"to encode texts ({exc}).",
                retryable=False,
            ) from exc

        selected_device = self._select_device(torch)
        resolved_dtype_name = resolve_model_dtype(
            self._configured_model_dtype, device=selected_device
        )
        self.model_dtype = resolved_dtype_name
        torch_dtype = self._resolve_torch_dtype(torch, resolved_dtype_name)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
                padding_side="left",
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            try:
                model = AutoModel.from_pretrained(
                    self.model_name,
                    revision=self.revision,
                    trust_remote_code=False,
                    local_files_only=self.local_files_only,
                    dtype=torch_dtype,
                )
            except TypeError:
                model = AutoModel.from_pretrained(
                    self.model_name,
                    revision=self.revision,
                    trust_remote_code=False,
                    local_files_only=self.local_files_only,
                    torch_dtype=torch_dtype,
                )

            if getattr(model, "config", None) is not None:
                model.config.use_cache = False

            model.eval()
            try:
                model.to(device=selected_device, dtype=torch_dtype)
            except RuntimeError as move_exc:
                if (
                    self.allow_cpu_fallback
                    and _normalize_device_type(selected_device) != "cpu"
                    and _is_oom_error(move_exc)
                ):
                    _empty_device_cache(torch, selected_device)
                    gc.collect()
                    selected_device = "cpu"
                    self.cpu_fallback_triggered = True
                    cpu_dtype_name = (
                        resolve_model_dtype(self._configured_model_dtype, device="cpu")
                        if self._configured_model_dtype not in {"float16", "bfloat16"}
                        else "float32"
                    )
                    model.to(
                        device="cpu",
                        dtype=self._resolve_torch_dtype(torch, cpu_dtype_name),
                    )
                else:
                    raise
        except UDLException:
            raise
        except Exception as exc:
            raise UDLException(
                "embedding_model_load_failed",
                f"Failed to load embedding model '{self.model_name}' "
                f"(revision {self.revision}): {exc}",
                retryable=False,
            ) from exc

        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size is not None and int(hidden_size) != self.dimension:
            raise UDLException(
                "embedding_dimension_mismatch",
                f"Loaded model hidden_size {hidden_size} does not match "
                f"configured dimension {self.dimension}.",
                retryable=False,
            )
        commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
        if commit_hash:
            self._resolved_revision = str(commit_hash)

        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = selected_device

    def _pad_token_id(self) -> int:
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        return 0

    def _tokenize_untruncated(
        self, texts: Sequence[str]
    ) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        """Tokenize `texts` without truncation (`truncation=False`, `padding=False`).

        Returns `(token_id_lists, attention_mask_lists_or_None)`.
        """
        try:
            tokenized = self._tokenizer(
                list(texts),
                padding=False,
                truncation=False,
            )
        except TypeError:
            tokenized = self._tokenizer(
                list(texts),
                padding=True,
                truncation=False,
                max_length=self.max_length,
            )

        raw_ids = tokenized["input_ids"]
        raw_masks = tokenized.get("attention_mask")

        id_lists: List[List[int]] = []
        mask_lists: Optional[List[List[int]]] = [] if raw_masks is not None else None
        for idx, seq in enumerate(raw_ids):
            seq_list = seq.tolist() if hasattr(seq, "tolist") else [int(x) for x in seq]
            if not seq_list:
                seq_list = [self._pad_token_id()]
            id_lists.append(seq_list)
            if mask_lists is not None and raw_masks is not None:
                m_seq = raw_masks[idx]
                m_list = (
                    m_seq.tolist()
                    if hasattr(m_seq, "tolist")
                    else [int(x) for x in m_seq]
                )
                if len(m_list) != len(seq_list):
                    m_list = [1] * len(seq_list)
                mask_lists.append(m_list)
        return id_lists, mask_lists

    def _forward_padded_batch(
        self,
        padded_ids: List[List[int]],
        padded_masks: List[List[int]],
        torch_module: Any,
    ) -> List[List[float]]:
        """Execute a single model forward pass on padded token IDs and masks."""
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            try:
                cfg.use_cache = False
            except Exception:
                pass

        if torch_module is not None and hasattr(self._model, "parameters"):
            target_device = self._resolved_device or "cpu"
            encoded = {
                "input_ids": torch_module.tensor(
                    padded_ids, dtype=torch_module.long, device=target_device
                ),
                "attention_mask": torch_module.tensor(
                    padded_masks, dtype=torch_module.long, device=target_device
                ),
            }
            ctx_factory = getattr(
                torch_module, "inference_mode", torch_module.no_grad
            )
            with ctx_factory():
                try:
                    outputs = self._model(**encoded, use_cache=False)
                except TypeError:
                    outputs = self._model(**encoded)
                last_hidden = (
                    outputs.last_hidden_state
                    if hasattr(outputs, "last_hidden_state")
                    else outputs[0]
                )
                pooled = last_valid_token_pool(
                    last_hidden, encoded["attention_mask"]
                )
                pooled_f32 = pooled.to(dtype=torch_module.float32)
                normalized = torch_module.nn.functional.normalize(
                    pooled_f32, p=2, dim=1
                )
                raw_vectors = normalized.cpu().tolist()
            del outputs, last_hidden, pooled, pooled_f32, normalized, encoded
            self._subbatch_count += 1
            max_seq_len = len(padded_ids[0]) if padded_ids else 0
            if _normalize_device_type(target_device) == "mps" and (
                max_seq_len >= 512 or self._subbatch_count % 16 == 0
            ):
                _empty_device_cache(torch_module, target_device)
            return raw_vectors

        encoded_py: Dict[str, Any] = {
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
        }
        try:
            outputs = self._model(**encoded_py, use_cache=False)
        except TypeError:
            outputs = self._model(**encoded_py)
        last_hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        pooled = last_valid_token_pool(
            last_hidden, encoded_py["attention_mask"]
        )
        return [l2_normalize(vec) for vec in pooled]

    def _fallback_model_to_cpu(self, torch_module: Any) -> None:
        prev_device = self._resolved_device or self.device or "cpu"
        _empty_device_cache(torch_module, prev_device)
        gc.collect()
        if hasattr(self._model, "to"):
            if torch_module is not None and hasattr(torch_module, self.model_dtype):
                # Keep the precision recorded in the embedding fingerprint.
                # Changing to float32 here would silently mix numerical profiles
                # inside one cached index after an accelerator OOM.
                self._model.to(device="cpu", dtype=getattr(torch_module, self.model_dtype))
            else:
                self._model.to("cpu")
        self._resolved_device = "cpu"
        self.cpu_fallback_triggered = True

    def _encode_window_subbatch(
        self,
        window_ids: Sequence[Sequence[int]],
        window_masks: Sequence[Optional[Sequence[int]]],
        torch_module: Any,
    ) -> List[List[float]]:
        """Left-pad a subbatch of token windows, run model forward, and pool + L2-normalize.

        On accelerator OOM, automatically splits multi-item subbatches into
        single-window microbatches and falls back to CPU at the configured precision if needed,
        never truncating or dropping any input tokens.
        """
        max_w_len = max(len(w) for w in window_ids)
        pad_id = self._pad_token_id()
        padding_side = str(getattr(self._tokenizer, "padding_side", "left")).lower()

        padded_ids: List[List[int]] = []
        padded_masks: List[List[int]] = []
        for w_ids, w_mask in zip(window_ids, window_masks):
            pad_len = max_w_len - len(w_ids)
            base_mask = (
                [int(x) for x in w_mask]
                if w_mask is not None and len(w_mask) == len(w_ids)
                else [1] * len(w_ids)
            )
            if padding_side == "right":
                padded_ids.append(list(w_ids) + [pad_id] * pad_len)
                padded_masks.append(base_mask + [0] * pad_len)
            else:
                padded_ids.append([pad_id] * pad_len + list(w_ids))
                padded_masks.append([0] * pad_len + base_mask)

        try:
            raw_vectors = self._forward_padded_batch(
                padded_ids, padded_masks, torch_module
            )
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            active_device = self._resolved_device or self.device or "cpu"
            _empty_device_cache(torch_module, active_device)
            gc.collect()

            # Step 1: If subbatch had multiple windows, split into single-window calls
            if len(window_ids) > 1:
                split_vecs: List[List[float]] = []
                for single_ids, single_mask in zip(window_ids, window_masks):
                    split_vecs.extend(
                        self._encode_window_subbatch(
                            [single_ids], [single_mask], torch_module
                        )
                    )
                return split_vecs

            # Step 2: Single window OOM -> CPU at the configured precision.
            if (
                self.allow_cpu_fallback
                and _normalize_device_type(active_device) != "cpu"
            ):
                self._fallback_model_to_cpu(torch_module)
                try:
                    raw_vectors = self._forward_padded_batch(
                        padded_ids, padded_masks, torch_module
                    )
                except RuntimeError as cpu_exc:
                    if _is_oom_error(cpu_exc):
                        raise UDLException(
                            "embedding_oom",
                            f"Out of memory encoding {max_w_len}-token window even after CPU fallback "
                            f"(batch_size={len(window_ids)}, model_dtype={self.model_dtype}). "
                            "Reduce max_length or batch_size; content was not truncated.",
                            retryable=True,
                        ) from cpu_exc
                    raise
            else:
                raise UDLException(
                    "embedding_oom",
                    f"Out of memory encoding {max_w_len}-token window on device '{active_device}' "
                    f"(batch_size={len(window_ids)}, model_dtype={self.model_dtype}). "
                    "Reduce batch_size/max_length or enable CPU fallback; content was not truncated.",
                    retryable=True,
                ) from exc

        out_vectors: List[List[float]] = []
        for vec in raw_vectors:
            norm_vec = l2_normalize(vec)
            if len(norm_vec) != self.dimension:
                raise UDLException(
                    "embedding_dimension_mismatch",
                    f"Encoded vector has dimension {len(norm_vec)}, "
                    f"expected {self.dimension}.",
                    retryable=False,
                )
            out_vectors.append(norm_vec)
        return out_vectors

    def _encode_formatted(
        self, formatted_texts: Sequence[str], *, is_query: bool = False
    ) -> List[List[float]]:
        if isinstance(formatted_texts, (str, bytes)):
            raise TypeError("Expected a sequence of strings, not a single string")
        items = [str(t) for t in formatted_texts]
        if not items:
            return []

        self._ensure_loaded()

        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        # Tokenize in manageable chunks without truncation so no tail tokens are ever silently lost
        all_doc_windows: List[List[Tuple[List[int], Optional[List[int]]]]] = []
        tok_chunk_size = max(16, self.batch_size * 8)

        for start in range(0, len(items), tok_chunk_size):
            sub_items = items[start : start + tok_chunk_size]
            id_lists, mask_lists = self._tokenize_untruncated(sub_items)
            for idx, seq_ids in enumerate(id_lists):
                seq_mask = mask_lists[idx] if mask_lists is not None else None
                if is_query and len(seq_ids) > self.max_length:
                    raise UDLException(
                        "query_too_long",
                        f"Query length ({len(seq_ids)} tokens) exceeds max_length ({self.max_length}).",
                        retryable=False,
                    )
                if len(seq_ids) <= self.max_length:
                    all_doc_windows.append([(seq_ids, seq_mask)])
                else:
                    id_windows = split_token_windows(
                        seq_ids,
                        max_length=self.max_length,
                        window_overlap=self.window_overlap,
                    )
                    if seq_mask is not None and len(seq_mask) == len(seq_ids):
                        mask_windows = split_token_windows(
                            seq_mask,
                            max_length=self.max_length,
                            window_overlap=self.window_overlap,
                        )
                    else:
                        mask_windows = [None] * len(id_windows)
                    all_doc_windows.append(list(zip(id_windows, mask_windows)))

        # Flatten all windows while tracking which original item each window belongs to
        flat_windows: List[Tuple[int, List[int], Optional[List[int]]]] = []
        for doc_wins in all_doc_windows:
            for w_ids, w_mask in doc_wins:
                flat_idx = len(flat_windows)
                flat_windows.append((flat_idx, w_ids, w_mask))

        # Bucket/batch windows safely when batch_size > 1 so long windows do not
        # force short sequences to pad to max_length or exceed conservative token budgets.
        if self.batch_size > 1 and any(len(w_ids) > 512 for _, w_ids, _ in flat_windows):
            ordered_windows = sorted(
                flat_windows, key=lambda item: (len(item[1]), item[0])
            )
        else:
            ordered_windows = flat_windows

        # Conservative token budget: any window > 512 tokens runs alone (batch=1)
        # to keep SDPA attention memory minimal on MPS/CPU.
        max_batch_tokens = max(self.max_length, min(self.batch_size * 512, 1024))
        encoded_by_flat_idx: Dict[int, List[float]] = {}

        cur_indices: List[int] = []
        cur_ids: List[List[int]] = []
        cur_masks: List[Optional[List[int]]] = []
        cur_max_len = 0

        def _flush_subbatch() -> None:
            if not cur_ids:
                return
            vecs = self._encode_window_subbatch(cur_ids, cur_masks, torch)
            for f_idx, vec in zip(cur_indices, vecs):
                encoded_by_flat_idx[f_idx] = vec
            cur_indices.clear()
            cur_ids.clear()
            cur_masks.clear()

        for f_idx, w_ids, w_mask in ordered_windows:
            w_len = len(w_ids)
            next_max_len = max(cur_max_len, w_len)
            next_count = len(cur_ids) + 1
            if cur_ids and (
                next_count > self.batch_size
                or next_max_len > 512
                or next_count * next_max_len > max_batch_tokens
            ):
                _flush_subbatch()
                cur_max_len = 0
            cur_indices.append(f_idx)
            cur_ids.append(w_ids)
            cur_masks.append(w_mask)
            cur_max_len = max(cur_max_len, w_len)

        _flush_subbatch()

        # Reassemble per-document window vectors and aggregate into 1 vector per chunk
        results: List[List[float]] = []
        cursor = 0
        for doc_wins in all_doc_windows:
            num_w = len(doc_wins)
            doc_vecs = [encoded_by_flat_idx[cursor + i] for i in range(num_w)]
            cursor += num_w
            agg_vec = aggregate_window_vectors(
                doc_vecs, aggregation=self.window_aggregation
            )
            if len(agg_vec) != self.dimension:
                raise UDLException(
                    "embedding_dimension_mismatch",
                    f"Aggregated vector has dimension {len(agg_vec)}, "
                    f"expected {self.dimension}.",
                    retryable=False,
                )
            results.append(agg_vec)

        return results

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode document texts into L2-normalized float32 vectors."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single str")
        formatted = [self.format_document(t) for t in texts]
        with self._inference_lock:
            return self._encode_formatted(formatted, is_query=False)

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode search queries with the instruction prefix into L2-normalized vectors."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single str")
        formatted = [self.format_query(t) for t in texts]
        with self._inference_lock:
            return self._encode_formatted(formatted, is_query=True)

    def runtime_smoke_check(
        self,
        sample_text: str = "Gradient descent updates parameters using the loss gradient.",
    ) -> Dict[str, Any]:
        """Run a lightweight single-document and single-query runtime verification."""
        doc_vec = self.encode_documents([sample_text])[0]
        query_vec = self.encode_queries([sample_text])[0]
        use_cache_flag = getattr(getattr(self._model, "config", None), "use_cache", False)
        return {
            "loaded": self.is_loaded,
            "device": self._resolved_device,
            "model_dtype": self.model_dtype,
            "use_cache": bool(use_cache_flag),
            "dimension": len(doc_vec),
            "doc_norm": math.sqrt(sum(v * v for v in doc_vec)),
            "query_norm": math.sqrt(sum(v * v for v in query_vec)),
            "cosine": cosine_similarity(doc_vec, query_vec),
            "cpu_fallback_triggered": self.cpu_fallback_triggered,
        }
