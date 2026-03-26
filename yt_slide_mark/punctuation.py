"""Punctuation restoration using ONNX models from 1-800-BAD-CODE.

Lightweight reimplementation of punctuators inference pipeline using only
onnxruntime + sentencepiece + numpy (no torch, no huggingface-hub).

Supports three models selected by language:
  - pcs_en: English only (default for -l en)
  - pcs_47lang: 47 languages including ru, uk, de, fr, zh, ar, ja, ko...
  - pcs_romance: ca, es, fr, it, pt, ro (lighter alternative for Romance)

Both SPE tokenizer and ONNX model are downloaded on first use and cached
in %LOCALAPPDATA%/yt-slide-mark (Windows) or ~/.cache/yt-slide-mark (Linux/macOS).
"""

import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/{repo}/resolve/main/{file}"


@dataclass
class _ModelConfig:
    repo: str
    spe_file: str
    onnx_file: str
    max_length: int
    pre_labels: list[str]
    post_labels: list[str]


_MODELS = {
    "pcs_en": _ModelConfig(
        repo="1-800-BAD-CODE/punct_cap_seg_en",
        spe_file="spe_32k_lc_en.model",
        onnx_file="punct_cap_seg_en.onnx",
        max_length=256,
        pre_labels=["<NULL>", "\u00bf"],
        post_labels=["<NULL>", "<ACRONYM>", ".", ",", "?"],
    ),
    "pcs_47lang": _ModelConfig(
        repo="1-800-BAD-CODE/punct_cap_seg_47_language",
        spe_file="spe_unigram_64k_lowercase_47lang.model",
        onnx_file="punct_cap_seg_47lang.onnx",
        max_length=128,
        pre_labels=["<NULL>", "\u00bf"],
        post_labels=["<NULL>", ".", ",", "?", "\uff1f", "\uff0c", "\u3002",
                      "\u3001", "\u30fb", "\u0964", "\u061f", "\u060c", ";",
                      "\u1362", "\u1363", "\u1367"],
    ),
    "pcs_romance": _ModelConfig(
        repo="1-800-BAD-CODE/punctuation_fullstop_truecase_romance",
        spe_file="sp.model",
        onnx_file="model.onnx",
        max_length=256,
        pre_labels=["<NULL>", "\u00bf"],
        post_labels=["<NULL>", "<ACRONYM>", ".", ",", "?"],
    ),
}

# Language → model mapping
_LANG_TO_MODEL = {"en": "pcs_en"}
for _lang in ["ca", "es", "fr", "it", "pt", "ro"]:
    _LANG_TO_MODEL[_lang] = "pcs_romance"
for _lang in [
    "af", "am", "ar", "bg", "bn", "de", "el", "et", "fa", "fi",
    "gu", "hi", "hr", "hu", "id", "is", "ja", "kk", "kn", "ko",
    "ky", "lt", "lv", "mk", "ml", "mr", "nl", "or", "pa", "pl",
    "ps", "ru", "rw", "so", "sr", "sw", "ta", "te", "tr", "uk", "zh",
]:
    _LANG_TO_MODEL[_lang] = "pcs_47lang"

_NULL = "<NULL>"
_ACRONYM = "<ACRONYM>"
_OVERLAP = 16

_loaded: dict[str, "_LitePunctuator"] = {}


def _cache_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    d = os.path.join(base, "yt-slide-mark")
    os.makedirs(d, exist_ok=True)
    return d


def _download_file(repo: str, filename: str) -> str:
    """Download a file from HuggingFace if not cached. Returns local path."""
    cache = _cache_dir()
    # Use repo slug + filename to avoid collisions between models
    slug = repo.replace("/", "--")
    local_dir = os.path.join(cache, slug)
    os.makedirs(local_dir, exist_ok=True)
    path = os.path.join(local_dir, filename)

    if os.path.exists(path):
        return path

    import requests

    url = _HF_BASE.format(repo=repo, file=filename)
    log.info("Downloading %s/%s …", repo.split("/")[-1], filename)
    tmp_path = path + ".tmp"
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    sys.stderr.write(f"\r  {downloaded >> 20}/{total >> 20} MB ({pct:.0f}%)")
                    sys.stderr.flush()
    if total:
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()
    os.replace(tmp_path, path)
    log.info("Cached: %s", path)
    return path


class _LitePunctuator:
    """Minimal punctuation/capitalization/segmentation using ONNX + SentencePiece."""

    def __init__(self, cfg: _ModelConfig, providers: list[str]):
        from sentencepiece import SentencePieceProcessor
        import onnxruntime as ort

        spe_path = _download_file(cfg.repo, cfg.spe_file)
        onnx_path = _download_file(cfg.repo, cfg.onnx_file)

        self._sp = SentencePieceProcessor(spe_path)
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._bos = self._sp.bos_id()
        self._eos = self._sp.eos_id()
        self._pad = self._sp.pad_id()
        self._cfg = cfg

    def infer(self, texts: list[str]) -> list[list[str]]:
        cfg = self._cfg
        max_tok = cfg.max_length - 2

        # Tokenize and split into overlapping segments
        segments = []
        for batch_idx, text in enumerate(texts):
            ids = self._sp.EncodeAsIds(text)
            start = 0
            input_idx = 0
            while start < len(ids):
                adj = start - (0 if input_idx == 0 else _OVERLAP)
                stop = adj + max_tok
                segments.append((ids[adj:stop], batch_idx, input_idx))
                start = stop
                input_idx += 1

        if not segments:
            return [[] for _ in texts]

        lengths = [len(s[0]) + 2 for s in segments]
        max_len = max(lengths)
        n = len(segments)

        input_ids = np.full((n, max_len), self._pad, dtype=np.int64)
        batch_indices = [s[1] for s in segments]
        input_indices = [s[2] for s in segments]

        for i, (ids, _, _) in enumerate(segments):
            seq = [self._bos] + ids + [self._eos]
            input_ids[i, :len(seq)] = seq

        pre_preds, post_preds, cap_preds, seg_preds = self._session.run(
            None, {"input_ids": input_ids}
        )

        collectors = [_Collector(self._sp, _OVERLAP) for _ in texts]
        for i in range(n):
            length = lengths[i]
            bidx = batch_indices[i]
            iidx = input_indices[i]
            s_ids = input_ids[i, 1:length - 1].tolist()
            s_pre = pre_preds[i, 1:length - 1].tolist()
            s_post = post_preds[i, 1:length - 1].tolist()
            s_cap = cap_preds[i, 1:length - 1].tolist()
            s_sbd = seg_preds[i, 1:length - 1].tolist()
            pre_tokens = [cfg.pre_labels[x] if cfg.pre_labels[x] != _NULL else None for x in s_pre]
            post_tokens = [cfg.post_labels[x] if cfg.post_labels[x] != _NULL else None for x in s_post]
            collectors[bidx].collect(s_ids, pre_tokens, post_tokens, s_cap, s_sbd, iidx)

        return [c.produce() for c in collectors]


class _Collector:
    """Reassembles punctuated text from model predictions."""

    def __init__(self, sp, overlap: int):
        self._sp = sp
        self._overlap = overlap
        self._segments: dict[int, tuple] = {}

    def collect(self, ids, pre_preds, post_preds, cap_preds, sbd_preds, idx):
        self._segments[idx] = (ids, pre_preds, post_preds, cap_preds, sbd_preds)

    def produce(self) -> list[str]:
        all_ids, all_pre, all_post, all_cap, all_sbd = [], [], [], [], []
        for i in range(len(self._segments)):
            ids, pre, post, cap, sbd = self._segments[i]
            start = 0
            stop = len(ids)
            if i > 0:
                start += self._overlap // 2
            if i < len(self._segments) - 1:
                stop -= self._overlap // 2
            all_ids.extend(ids[start:stop])
            all_pre.extend(pre[start:stop])
            all_post.extend(post[start:stop])
            all_cap.extend(cap[start:stop])
            all_sbd.extend(sbd[start:stop])

        tokens = [self._sp.IdToPiece(x) for x in all_ids]
        sentences: list[str] = []
        chars: list[str] = []

        for ti, token in enumerate(tokens):
            if token.startswith("\u2581") and chars:
                chars.append(" ")
            cs = 1 if token.startswith("\u2581") else 0
            for ci, ch in enumerate(token[cs:], start=cs):
                if ci == cs and all_pre[ti] is not None:
                    chars.append(all_pre[ti])
                if all_cap[ti][ci]:
                    ch = ch.upper()
                chars.append(ch)
                label = all_post[ti]
                if label == _ACRONYM:
                    chars.append(".")
                elif ci == len(token) - 1 and label is not None:
                    chars.append(label)
                if ci == len(token) - 1 and all_sbd[ti]:
                    sentences.append("".join(chars))
                    chars = []

        if chars:
            sentences.append("".join(chars))
        return sentences


def _get_providers() -> list[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_model(language: str) -> _LitePunctuator:
    model_key = _LANG_TO_MODEL.get(language, "pcs_47lang")
    if model_key not in _loaded:
        cfg = _MODELS[model_key]
        log.info("Loading punctuation model %s (first use for this language)…", model_key)
        _loaded[model_key] = _LitePunctuator(cfg, _get_providers())
        log.info("Punctuation model loaded")
    return _loaded[model_key]


def supported_languages() -> list[str]:
    """Return list of supported language codes."""
    return sorted(_LANG_TO_MODEL.keys())


def punctuate_texts(texts: list[str], language: str = "en") -> list[str]:
    """Restore punctuation and capitalization for a batch of texts."""
    if not texts:
        return []

    model = _get_model(language)
    results = model.infer(texts)

    out = []
    for result in results:
        if isinstance(result, list):
            out.append(" ".join(result))
        else:
            out.append(str(result))
    return out
