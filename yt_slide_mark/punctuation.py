import logging

import onnxruntime as ort

log = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is None:
        log.info("Loading punctuation model (first use)…")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            log.info("ONNX: using CUDA + CPU")
        else:
            providers = ["CPUExecutionProvider"]
            log.info("ONNX: using CPU only")

        # Monkey-patch InferenceSession to inject our provider list,
        # since punctuators doesn't expose a providers parameter.
        _orig_init = ort.InferenceSession.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs["providers"] = providers
            _orig_init(self, *args, **kwargs)

        ort.InferenceSession.__init__ = _patched_init
        try:
            from punctuators.models import PunctCapSegModelONNX
            _model = PunctCapSegModelONNX.from_pretrained("pcs_en")
        finally:
            ort.InferenceSession.__init__ = _orig_init
        log.info("Punctuation model loaded")
    return _model


def punctuate_texts(texts: list[str]) -> list[str]:
    """Restore punctuation and capitalization for a batch of texts.

    Uses punctuators pcs_en (ONNX, CPU). Model is loaded lazily on first call.
    """
    if not texts:
        return []

    model = _load_model()
    results = model.infer(texts)

    out = []
    for result in results:
        # Each result is a list of sentences; join them back
        if isinstance(result, list):
            out.append(" ".join(result))
        else:
            out.append(str(result))
    return out
