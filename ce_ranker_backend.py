"""Qwen CE backend extracted from PROBE_wt_ce/join_cross_encoder.py.

Only the predicate-aware Qwen backend is included. Score is yes-minus-no logit.
"""
from typing import Dict, List
import numpy as np
QWEN_SYSTEM = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'

def _forward_last_token(model, enc):
    """Last-token hidden state of one batch, with position ids that never leak across batches.

    Qwen3-VL caches the batch's rope deltas on the model object and, for a batch without images,
    reuses the cached ones: after a 24-pair image batch a 32-pair text batch crashes, and a text
    batch of the same size silently gets another batch's positions. Clear the cache every call and,
    without images, pass positions from the attention mask so left padding does not shift them.
    """
    import torch
    model.rope_deltas = None
    kwargs = dict(enc)
    if kwargs.get("pixel_values") is None:
        am = kwargs["attention_mask"]
        pos = (am.long().cumsum(-1) - 1).clamp(min=0)
        kwargs["position_ids"] = pos.unsqueeze(0).expand(3, -1, -1)
    return model(**kwargs).last_hidden_state[:, -1]

def _linear_patch_embed(visual) -> None:
    """Replace the vision tower's Conv3d patch embedding with the equivalent linear map.

    The kernel equals the stride, so the convolution is a matmul over flattened patches. On the RTX PRO
    6000 (Blackwell) the bf16 Conv3d took 8.8 s for 1,960 patches versus 0.05 ms as a linear layer, with
    outputs equal up to bf16 rounding (max |diff| 0.008): images ran at 0.4 pairs/s before this.
    """
    import torch.nn.functional as F
    pe = visual.patch_embed
    w = pe.proj.weight.reshape(pe.proj.weight.shape[0], -1)
    b = pe.proj.bias

    def forward(hidden_states):
        return F.linear(hidden_states.reshape(-1, w.shape[1]).to(w.dtype), w, b)

    pe.forward = forward

class _QwenVL:
    def __init__(self, device: str, model_path: str, max_pixels=262144, max_text_chars=2000):
        self.max_pixels = max_pixels
        self.max_text_chars = max_text_chars
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        lm = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=dtype).to(device).eval()
        self.model = lm.model
        _linear_patch_embed(self.model.visual)
        self.proc = AutoProcessor.from_pretrained(model_path, padding_side="left")
        vocab = self.proc.tokenizer.get_vocab()
        w = lm.lm_head.weight.data
        self.w = (w[vocab["yes"]] - w[vocab["no"]]).to(device=device, dtype=torch.float32)
        del lm.lm_head
        self._images: Dict[str, object] = {}

    def _image(self, path: str):
        from PIL import Image, ImageFile
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            raise ValueError('CE requires strict image decoding; run with '
                             'PIL.ImageFile.LOAD_TRUNCATED_IMAGES=False in an isolated process')
        im = self._images.get(path)
        if im is None:
            im = Image.open(path).convert("RGB")
            px = im.width * im.height
            s = 1.0
            if px > self.max_pixels:
                s = (self.max_pixels / px) ** 0.5
            elif px < 4096:
                s = (4096 / px) ** 0.5
            # The processor runs with do_resize=False, so both sides must be multiples of 32.
            w = max(32, int(im.width * s) // 32 * 32)
            h = max(32, int(im.height * s) // 32 * 32)
            if (w, h) != im.size:
                im = im.resize((w, h), Image.BICUBIC)
            self._images[path] = im
        return im

    def _side(self, row: dict, prefix: str, images: list) -> list:
        content = [{"type": "text", "text": prefix}]
        if row["image"]:
            content.append({"type": "image"})
            images.append(self._image(row["image"]))
        if row["text"]:
            content.append({"type": "text", "text": row["text"][:self.max_text_chars]})
        if len(content) == 1:
            content.append({"type": "text", "text": "NULL"})
        return content

    def encode(self, left: List[dict], right: List[dict], instruction: str, batch: int):
        import torch
        scores, hidden = [], []
        with torch.no_grad():
            for i in range(0, len(left), batch):
                texts, images = [], []
                for a, b in zip(left[i:i + batch], right[i:i + batch]):
                    user = [{"type": "text", "text": "<Instruct>: " + instruction}]
                    user += self._side(a, "<Query>:", images)
                    user += self._side(b, "\n<Document>:", images)
                    msgs = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
                            {"role": "user", "content": user}]
                    texts.append(self.proc.apply_chat_template(msgs, tokenize=False,
                                                               add_generation_prompt=True))
                enc = self.proc(text=texts, images=images or None, padding=True, return_tensors="pt",
                                do_resize=False).to(self.model.device)
                h = _forward_last_token(self.model, enc)
                scores.append((h.float() @ self.w).cpu().numpy())
                hidden.append(h.float().cpu().numpy())
        return np.concatenate(scores), np.concatenate(hidden)


class _JinaM0:
    """jina-reranker-m0 (Qwen2-VL-2B backbone): same interface as _QwenVL.

    The model reads ``**Document**:\\n<doc>\\n**Query**:\\n<query>`` followed by a score token (id 100)
    and scores the last-token hidden state with a two-layer MLP. We return that raw MLP output (before
    the model's sigmoid(x - 2.65) display normalisation, which is monotone and so leaves the order
    unchanged) and the same last-token hidden state as the representation. A side with both an image
    and text puts the image first; the model was trained on either one, so a mixed side is a
    documented extrapolation. The instruction (predicate) is the prompt prefix, as in _QwenVL.
    """

    SCORE_TOKEN_ID = 100

    def __init__(self, device: str, model_path: str, max_pixels=262144, max_text_chars=2000):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.max_text_chars = max_text_chars
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        m = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).to(device).eval()
        self.model = m
        # The score head runs in float32: in bf16 its output has ~3 significant digits, which ties
        # many candidates on the fixed sampling axis.
        import copy
        self.score_head = copy.deepcopy(m.score).float()
        self.proc = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=3136,
                                                  trust_remote_code=True, padding_side="left")
        self._images: Dict[str, object] = {}

    def _image(self, path: str):
        from PIL import Image, ImageFile
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            raise ValueError('CE requires strict image decoding; run with '
                             'PIL.ImageFile.LOAD_TRUNCATED_IMAGES=False in an isolated process')
        im = self._images.get(path)
        if im is None:
            im = Image.open(path).convert("RGB")
            self._images[path] = im
        return im

    def _side(self, row: dict, images: list) -> str:
        parts = []
        if row["image"]:
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
            images.append(self._image(row["image"]))
        if row["text"]:
            parts.append(row["text"][:self.max_text_chars])
        return "\n".join(parts) if parts else "NULL"

    def encode(self, left: List[dict], right: List[dict], instruction: str, batch: int):
        import torch
        scores, hidden = [], []
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(left), batch):
                texts, images = [], []
                for a, b in zip(left[i:i + batch], right[i:i + batch]):
                    # Images must be listed in prompt order: the Document side comes first.
                    doc_imgs, query_imgs = [], []
                    doc = self._side(b, doc_imgs)
                    query = self._side(a, query_imgs)
                    images += doc_imgs + query_imgs
                    texts.append(f"{instruction}\n**Document**:\n{doc}\n**Query**:\n{query}")
                enc = self.proc(text=texts, images=images or None, padding=True, return_tensors="pt")
                n = enc["input_ids"].size(0)
                tok = torch.full((n, 1), self.SCORE_TOKEN_ID, dtype=enc["input_ids"].dtype)
                enc["input_ids"] = torch.cat([enc["input_ids"], tok], dim=1)
                enc["attention_mask"] = torch.cat(
                    [enc["attention_mask"], torch.ones(n, 1, dtype=enc["attention_mask"].dtype)], dim=1)
                if "mm_token_type_ids" in enc:
                    enc["mm_token_type_ids"] = torch.cat(
                        [enc["mm_token_type_ids"], torch.zeros(n, 1, dtype=enc["mm_token_type_ids"].dtype)], dim=1)
                enc = enc.to(dev)
                # Cached rope deltas from a previous batch would shift this batch's positions.
                if hasattr(self.model, "model"):
                    self.model.model.rope_deltas = None
                out = self.model.model(**{k: v for k, v in enc.items()}, use_cache=False)
                h = out.last_hidden_state[:, -1]
                s = self.score_head(h.float()).squeeze(-1)
                scores.append(s.float().cpu().numpy())
                hidden.append(h.float().cpu().numpy())
        return np.concatenate(scores), np.concatenate(hidden)


BACKENDS = {"Qwen/Qwen3-VL-Reranker-2B": _QwenVL, "jinaai/jina-reranker-m0": _JinaM0}
SCORE_NAMES = {"Qwen/Qwen3-VL-Reranker-2B": "yes_minus_no_logit", "jinaai/jina-reranker-m0": "score_head_logit"}
