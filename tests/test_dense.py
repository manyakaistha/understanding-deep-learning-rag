"""Unit and contract tests for `src/dense.py`."""

import math
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.contracts import DEFAULT_EMBEDDING_MODEL, UDLException
from src.dense import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_MAX_LENGTH,
    DEFAULT_OVERFLOW_POLICY,
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_QUERY_OVERFLOW_POLICY,
    DEFAULT_WINDOW_AGGREGATION,
    DEFAULT_WINDOW_OVERLAP,
    LocalEmbedder,
    aggregate_window_vectors,
    canonical_profile_dict,
    compute_profile_fingerprint,
    cosine_similarity,
    embedding_cache_key,
    format_document_prompt,
    format_query_prompt,
    l2_normalize,
    last_valid_token_pool,
    pack_vector,
    resolve_model_dtype,
    split_token_windows,
    unpack_vector,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.call_kwargs = []
        self.padding_side = "left"
        self.pad_token_id = 0

    def __call__(self, texts, padding=False, truncation=False, max_length=4096):
        self.calls.append(list(texts))
        self.call_kwargs.append(
            {"padding": padding, "truncation": truncation, "max_length": max_length}
        )
        # Simulate variable-length token IDs and masks without truncating
        input_ids = []
        masks = []
        for text in texts:
            words = str(text).split()
            ids = [100 + ((hash(w) + i) % 900) for i, w in enumerate(words)] or [101]
            input_ids.append(ids)
            masks.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": masks}


class _DummyOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyModelConfig:
    def __init__(self, hidden_size=4, commit_hash=DEFAULT_EMBEDDING_REVISION, use_cache=True):
        self.hidden_size = hidden_size
        self._commit_hash = commit_hash
        self.use_cache = use_cache


class _DummyModel:
    def __init__(
        self,
        hidden_size=4,
        *,
        oom_on_batch_gt_1=False,
        oom_on_mps=False,
        always_oom=False,
    ) -> None:
        self.config = _DummyModelConfig(hidden_size=hidden_size, use_cache=True)
        self.seen_input_ids = []
        self.seen_use_cache = []
        self.to_calls = []
        self.current_device = "mps"
        self.oom_on_batch_gt_1 = oom_on_batch_gt_1
        self.oom_on_mps = oom_on_mps
        self.always_oom = always_oom

    def to(self, device=None, dtype=None):
        target = str(device) if device is not None else self.current_device
        self.current_device = target
        self.to_calls.append((target, str(dtype) if dtype is not None else None))
        return self

    def __call__(self, **encoded):
        input_ids = encoded["input_ids"]
        masks = encoded["attention_mask"]
        self.seen_use_cache.append(encoded.get("use_cache"))

        if self.always_oom:
            raise RuntimeError("MPS backend out of memory (simulated permanent OOM)")
        if self.oom_on_batch_gt_1 and len(input_ids) > 1:
            raise RuntimeError(
                "MPS backend out of memory (simulated multi-window subbatch OOM)"
            )
        if self.oom_on_mps and self.current_device.startswith("mps"):
            raise RuntimeError(
                "MPS backend out of memory (simulated single-window MPS OOM)"
            )

        batch_states = []
        for row_ids, row_mask in zip(input_ids, masks):
            valid_ids = [
                int(tid) for tid, m in zip(row_ids, row_mask) if int(m) > 0
            ]
            self.seen_input_ids.append(valid_ids)
            seq_states = []
            running = [0.0] * self.config.hidden_size
            for s_idx, (tid, m) in enumerate(zip(row_ids, row_mask)):
                if int(m) > 0:
                    slot = int(tid) % self.config.hidden_size
                    running[slot] += float((int(tid) % 29) + 1)
                    running[(slot + 1) % self.config.hidden_size] += float(s_idx + 1) * 0.25
                vec = list(running)
                if sum(v * v for v in vec) == 0.0:
                    vec[0] = 1.0
                seq_states.append(vec)
            batch_states.append(seq_states)
        return _DummyOutput(batch_states)


class DenseTests(unittest.TestCase):
    def test_concurrent_search_and_build_calls_share_one_inference_lane(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        embedder = LocalEmbedder(device="cpu")
        entered, release = threading.Event(), threading.Event()
        calls = []
        def forward(texts, is_query=False):
            calls.append(is_query)
            if len(calls) == 1:
                entered.set()
                self.assertTrue(release.wait(timeout=2))
            return [[1.0, 0.0]]
        with patch.object(embedder, "_encode_formatted", side_effect=forward):
            with ThreadPoolExecutor(max_workers=2) as executor:
                build = executor.submit(embedder.encode_documents, ["document"])
                self.assertTrue(entered.wait(timeout=2))
                query = executor.submit(embedder.encode_queries, ["question"])
                self.assertEqual(calls, [False])
                release.set()
                self.assertEqual(build.result(timeout=2), [[1.0, 0.0]])
                self.assertEqual(query.result(timeout=2), [[1.0, 0.0]])
        self.assertEqual(calls, [False, True])

    def test_lazy_initialization_and_profile_contract(self):
        embedder = LocalEmbedder()
        self.assertFalse(embedder.is_loaded)
        profile = embedder.profile
        self.assertEqual(profile["model_name"], DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(profile["revision"], DEFAULT_EMBEDDING_REVISION)
        self.assertEqual(profile["dimension"], DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(profile["pooling"], "last_valid_token")
        self.assertEqual(profile["normalize"], "l2")
        self.assertEqual(profile["dtype"], "float32")
        self.assertIn(profile["model_dtype"], {"float16", "bfloat16", "float32"})
        self.assertEqual(profile["max_length"], DEFAULT_MAX_LENGTH)
        self.assertEqual(profile["overflow_policy"], DEFAULT_OVERFLOW_POLICY)
        self.assertEqual(profile["window_overlap"], DEFAULT_WINDOW_OVERLAP)
        self.assertEqual(profile["window_aggregation"], DEFAULT_WINDOW_AGGREGATION)
        self.assertEqual(profile["query_overflow_policy"], DEFAULT_QUERY_OVERFLOW_POLICY)
        self.assertEqual(profile["query_instruction"], DEFAULT_QUERY_INSTRUCTION)
        self.assertIn("Instruct:", profile["query_prompt"])
        self.assertIn("Query:", profile["query_prompt"])
        self.assertEqual(len(embedder.fingerprint), 64)
        self.assertFalse(embedder.is_loaded)

        # Device-aware default model_dtype resolution
        self.assertEqual(resolve_model_dtype(None, device="mps"), "float16")
        self.assertEqual(resolve_model_dtype(None, device="cuda"), "float16")
        self.assertEqual(resolve_model_dtype(None, device="cpu"), "float32")
        self.assertEqual(resolve_model_dtype("float16", device="cpu"), "float16")
        self.assertEqual(resolve_model_dtype("float32", device="mps"), "float32")

        emb_mps = LocalEmbedder(device="mps")
        emb_cpu = LocalEmbedder(device="cpu")
        self.assertEqual(emb_mps.profile["model_dtype"], "float16")
        self.assertEqual(emb_cpu.profile["model_dtype"], "float32")

    def test_query_and_document_formatting(self):
        q_formatted = format_query_prompt("What is backpropagation?")
        self.assertEqual(
            q_formatted,
            f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery:What is backpropagation?",
        )
        d_formatted = format_document_prompt("Backpropagation computes gradients.")
        self.assertEqual(d_formatted, "Backpropagation computes gradients.")

    def test_last_valid_token_pool_left_and_right_padding(self):
        hidden = [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [9.0, 9.0]],  # right-padded: last valid at idx 2 -> [3.0, 0.0]
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 5.0]],  # left-padded: last valid at idx 3 -> [0.0, 5.0]
        ]
        mask = [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
        ]
        pooled = last_valid_token_pool(hidden, mask)
        self.assertEqual(pooled, [[3.0, 0.0], [0.0, 5.0]])

    def test_vector_packing_normalization_and_cosine(self):
        raw = [3.0, 4.0, 0.0, 0.0]
        normed = l2_normalize(raw)
        self.assertAlmostEqual(normed[0], 0.6, places=6)
        self.assertAlmostEqual(normed[1], 0.8, places=6)
        self.assertAlmostEqual(sum(x * x for x in normed), 1.0, places=6)

        blob = pack_vector(normed, expected_dim=4)
        self.assertEqual(len(blob), 16)
        unpacked = unpack_vector(blob, expected_dim=4)
        for a, b in zip(normed, unpacked):
            self.assertAlmostEqual(a, b, places=6)

        self.assertAlmostEqual(cosine_similarity(normed, unpacked), 1.0, places=6)
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6
        )

        with self.assertRaises(UDLException):
            l2_normalize([0.0, 0.0])
        with self.assertRaises(UDLException):
            pack_vector([float("nan"), 1.0])
        with self.assertRaises(UDLException):
            unpack_vector(b"bad", expected_dim=4)
        with self.assertRaises(UDLException):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_embedding_cache_key_and_profile_fingerprint_sensitivity(self):
        emb_a = LocalEmbedder(device="mps", model_dtype="float16")
        emb_b = LocalEmbedder(device="mps", model_dtype="float16", revision="different_revision_sha")
        emb_c = LocalEmbedder(device="mps", model_dtype="float16", query_instruction="Different query instruction")
        emb_d = LocalEmbedder(device="mps", model_dtype="float16", window_overlap=64)
        emb_e = LocalEmbedder(device="mps", model_dtype="float16", overflow_policy="custom_window")
        emb_fp32 = LocalEmbedder(device="cpu", model_dtype="float32")

        self.assertNotEqual(emb_a.fingerprint, emb_d.fingerprint)
        self.assertNotEqual(emb_a.fingerprint, emb_e.fingerprint)
        self.assertNotEqual(emb_a.fingerprint, emb_fp32.fingerprint)

        # Verify extra identity fields in custom profiles are preserved in canonical_profile_dict
        custom_prof_1 = dict(emb_a.profile, custom_identity_tag="v1")
        custom_prof_2 = dict(emb_a.profile, custom_identity_tag="v2")
        self.assertEqual(canonical_profile_dict(custom_prof_1)["custom_identity_tag"], "v1")
        self.assertNotEqual(
            compute_profile_fingerprint(custom_prof_1),
            compute_profile_fingerprint(custom_prof_2),
        )
        self.assertNotEqual(
            embedding_cache_key("Gradient descent", custom_prof_1),
            embedding_cache_key("Gradient descent", custom_prof_2),
        )

        key_doc_a = embedding_cache_key("Gradient descent", emb_a.profile)
        key_doc_a2 = embedding_cache_key("Gradient descent", emb_a.profile)
        key_doc_diff_text = embedding_cache_key("Stochastic gradient descent", emb_a.profile)
        key_doc_diff_rev = embedding_cache_key("Gradient descent", emb_b.profile)
        key_doc_diff_inst = embedding_cache_key("Gradient descent", emb_c.profile)
        key_doc_diff_overlap = embedding_cache_key("Gradient descent", emb_d.profile)
        key_doc_diff_dtype = embedding_cache_key("Gradient descent", emb_fp32.profile)
        key_query_a = embedding_cache_key("Gradient descent", emb_a.profile, is_query=True)

        self.assertEqual(key_doc_a, key_doc_a2)
        self.assertNotEqual(key_doc_a, key_doc_diff_text)
        self.assertNotEqual(key_doc_a, key_doc_diff_rev)
        self.assertNotEqual(key_doc_a, key_doc_diff_inst)
        self.assertNotEqual(key_doc_a, key_doc_diff_overlap)
        self.assertNotEqual(key_doc_a, key_doc_diff_dtype)
        self.assertNotEqual(key_doc_a, key_query_a)

    def test_encode_documents_and_queries_with_injected_test_double(self):
        tokenizer = _DummyTokenizer()
        model = _DummyModel(hidden_size=4)
        self.assertTrue(model.config.use_cache)
        embedder = LocalEmbedder(
            dimension=4,
            batch_size=2,
            tokenizer=tokenizer,
            model=model,
        )
        self.assertTrue(embedder.is_loaded)
        self.assertFalse(model.config.use_cache)

        doc_vecs = embedder.encode_documents(["doc one", "doc two"])
        self.assertEqual(len(doc_vecs), 2)
        self.assertEqual(tokenizer.calls[0], ["doc one", "doc two"])
        self.assertFalse(tokenizer.call_kwargs[0]["truncation"])
        self.assertEqual(model.seen_use_cache[0], False)
        for vec in doc_vecs:
            self.assertEqual(len(vec), 4)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in vec)), 1.0, places=6)

        query_vecs = embedder.encode_queries(["what is attention?"])
        self.assertEqual(len(query_vecs), 1)
        self.assertTrue(tokenizer.calls[1][0].startswith("Instruct: "))
        self.assertIn("\nQuery:what is attention?", tokenizer.calls[1][0])

    def test_long_document_sliding_windows_preserve_all_tokens_and_tail_sensitivity(self):
        # Verify split_token_windows directly
        tokens_25 = list(range(1, 26))
        wins = split_token_windows(tokens_25, max_length=10, window_overlap=3)
        # Stride is 10 - 3 = 7: windows [0:10], [7:17], [14:24], [21:25]
        self.assertEqual(len(wins), 4)
        self.assertTrue(all(len(w) <= 10 for w in wins))
        # Overlap between consecutive windows is 3
        self.assertEqual(wins[0][-3:], wins[1][:3])
        self.assertEqual(wins[1][-3:], wins[2][:3])
        self.assertEqual(wins[2][-3:], wins[3][:3])
        # Every single input token is covered
        covered = [t for w in wins for t in w]
        self.assertEqual(set(covered), set(tokens_25))

        # Test with injected model double: max_length=8, window_overlap=2 (stride=6)
        tokenizer = _DummyTokenizer()
        model = _DummyModel(hidden_size=4)
        embedder = LocalEmbedder(
            dimension=4,
            max_length=8,
            window_overlap=2,
            batch_size=2,
            tokenizer=tokenizer,
            model=model,
        )

        prefix_words = [f"tok_{i}" for i in range(16)]  # 16 tokens = 2 windows before tail
        doc_tail_a = " ".join(prefix_words + ["tail_alpha", "tail_beta", "tail_gamma"])
        doc_tail_b = " ".join(prefix_words + ["tail_delta", "tail_epsilon", "tail_zeta"])

        model.seen_input_ids.clear()
        vecs = embedder.encode_documents([doc_tail_a, doc_tail_b])
        # Must preserve 1 vector per chunk
        self.assertEqual(len(vecs), 2)
        self.assertAlmostEqual(sum(x * x for x in vecs[0]), 1.0, places=6)
        self.assertAlmostEqual(sum(x * x for x in vecs[1]), 1.0, places=6)

        # Altering the tail after the first window MUST change the final document vector
        self.assertNotEqual(vecs[0], vecs[1])

        # Verify no input token from either document was lost across the model forward passes
        expected_ids_a = tokenizer([doc_tail_a])["input_ids"][0]
        expected_ids_b = tokenizer([doc_tail_b])["input_ids"][0]
        flattened_seen = {tid for seq in model.seen_input_ids for tid in seq}
        self.assertTrue(set(expected_ids_a).issubset(flattened_seen))
        self.assertTrue(set(expected_ids_b).issubset(flattened_seen))
        self.assertTrue(all(len(seq) <= 8 for seq in model.seen_input_ids))

        # Queries exceeding max_length must be rejected explicitly
        overlong_query = " ".join(f"q_{i}" for i in range(12))
        with self.assertRaises(UDLException) as ctx:
            embedder.encode_queries([overlong_query])
        self.assertEqual(ctx.exception.code, "query_too_long")

    @patch.dict("sys.modules", {"torch": SimpleNamespace(float16="torch.float16")})
    def test_oom_microbatch_split_cpu_fallback_and_actionable_error_without_truncation(self):
        # 1. Subbatch OOM when batch > 1 automatically splits to single-window microbatches
        tok_split = _DummyTokenizer()
        model_split = _DummyModel(hidden_size=4, oom_on_batch_gt_1=True)
        emb_split = LocalEmbedder(
            dimension=4,
            max_length=8,
            window_overlap=2,
            batch_size=4,
            device="mps",
            tokenizer=tok_split,
            model=model_split,
        )
        vecs_split = emb_split.encode_documents(
            ["alpha beta gamma", "delta epsilon zeta eta"]
        )
        self.assertEqual(len(vecs_split), 2)
        self.assertFalse(emb_split.cpu_fallback_triggered)

        # 2. CPU fallback preserves the precision in the cache fingerprint.
        tok_fb = _DummyTokenizer()
        model_fb = _DummyModel(hidden_size=4, oom_on_mps=True)
        emb_fb = LocalEmbedder(
            dimension=4,
            max_length=6,
            window_overlap=2,
            batch_size=1,
            device="mps",
            model_dtype="float16",
            allow_cpu_fallback=True,
            tokenizer=tok_fb,
            model=model_fb,
        )
        fp_before = emb_fb.fingerprint
        long_doc = " ".join(f"word_{i}" for i in range(11))
        expected_ids = tok_fb([long_doc])["input_ids"][0]
        model_fb.seen_input_ids.clear()

        vecs_fb = emb_fb.encode_documents([long_doc])
        self.assertEqual(len(vecs_fb), 1)
        self.assertTrue(emb_fb.cpu_fallback_triggered)
        self.assertEqual(emb_fb._resolved_device, "cpu")
        self.assertEqual(model_fb.current_device, "cpu")
        self.assertEqual(model_fb.to_calls[-1][1], "torch.float16")
        # Profile fingerprint remains stable across fallback
        self.assertEqual(emb_fb.fingerprint, fp_before)
        # Every token was encoded without truncation
        seen_ids = {tid for seq in model_fb.seen_input_ids for tid in seq}
        self.assertEqual(set(expected_ids), seen_ids)

        # 3. Unrecoverable OOM (or allow_cpu_fallback=False) raises actionable UDLException("embedding_oom")
        tok_fail = _DummyTokenizer()
        model_fail = _DummyModel(hidden_size=4, always_oom=True)
        emb_fail = LocalEmbedder(
            dimension=4,
            max_length=8,
            window_overlap=2,
            batch_size=1,
            device="mps",
            allow_cpu_fallback=True,
            tokenizer=tok_fail,
            model=model_fail,
        )
        with self.assertRaises(UDLException) as ctx:
            emb_fail.encode_documents(["alpha beta gamma"])
        self.assertEqual(ctx.exception.code, "embedding_oom")
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("not truncated", ctx.exception.message)

    @unittest.skipUnless(
        os.environ.get("UDL_RUN_REAL_EMBEDDER_CHECK") == "1",
        "Set UDL_RUN_REAL_EMBEDDER_CHECK=1 to run optional real Qwen3 CPU runtime check",
    )
    def test_optional_real_qwen_cpu_smoke_and_window_tail_fidelity(self):
        embedder = LocalEmbedder(
            device="cpu",
            model_dtype="float32",
            batch_size=1,
            max_length=64,
            window_overlap=16,
            local_files_only=True,
        )
        smoke = embedder.runtime_smoke_check(
            "Backpropagation computes the gradient of the loss with respect to parameters."
        )
        self.assertTrue(smoke["loaded"])
        self.assertEqual(smoke["device"], "cpu")
        self.assertEqual(smoke["model_dtype"], "float32")
        self.assertFalse(smoke["use_cache"])
        self.assertEqual(smoke["dimension"], DEFAULT_EMBEDDING_DIMENSION)
        self.assertAlmostEqual(smoke["doc_norm"], 1.0, places=5)
        self.assertAlmostEqual(smoke["query_norm"], 1.0, places=5)

        prefix = " ".join(["gradient descent optimization step"] * 25)
        doc_a = f"{prefix} tail_token_backpropagation_equations"
        doc_b = f"{prefix} tail_token_convolutional_filters"
        vec_a, vec_b = embedder.encode_documents([doc_a, doc_b])
        self.assertNotEqual(vec_a, vec_b)


if __name__ == "__main__":
    unittest.main()
