"""
意图文本 → 固定维向量 编码工具

设计说明（对应研究框架「VLM 语义锚点」）:
    1. Qwen2.5-VL（vLLM）负责从 FPV 图像生成**自然语言意图描述**（高层语义）
    2. 本模块负责将描述编码为**固定维度向量**，写入预处理 .pt，供 CoFE / PTINet 离线读取
    3. 默认使用 CLIP 文本编码器输出 512 维，与 config 中 intent_feature_dim=512 对齐

依赖（二选一，脚本启动时自动检测）:
    pip install transformers pillow
    或
    pip install sentence-transformers
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 将任意维 embedding 对齐到目标维度的确定性投影种子（同一文本多次运行结果一致）
_PROJECTION_SEED = 42


def _l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 归一化，便于与余弦相似度训练目标对齐。"""
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def _project_to_dim(vec: np.ndarray, out_dim: int) -> np.ndarray:
    """
    将任意长度向量投影到 out_dim（默认 512）。

    使用固定随机正交投影矩阵，保证：
        - 不引入可学习参数文件
        - 同一输入多次运行结果一致
    """
    in_dim = vec.shape[0]
    if in_dim == out_dim:
        return vec.astype(np.float32)
    rng = np.random.RandomState(_PROJECTION_SEED + in_dim * 1000 + out_dim)
    proj = rng.randn(in_dim, out_dim).astype(np.float32) / np.sqrt(in_dim)
    out = vec.astype(np.float32) @ proj
    return out


class IntentTextEncoder:
    """
    将 VLM 输出的意图文本编码为 intent_feature 向量。

    优先级:
        1. CLIP (ViT-B/32) 文本塔 → 512 维（推荐，与 README 一致）
        2. sentence-transformers MiniLM → 384 维再投影到 out_dim
    """

    def __init__(self, out_dim: int = 512, device: str = "cpu"):
        self.out_dim = out_dim
        self.device = device
        self._backend = None
        self._model = None
        self._processor = None
        self._init_backend()

    def _init_backend(self) -> None:
        # --- 方案 1: HuggingFace CLIP 文本塔（512 维，仅需 tokenizer + 文本前向）---
        try:
            import torch
            from transformers import CLIPModel, CLIPTokenizer

            model_name = "openai/clip-vit-base-patch32"
            logger.info("加载 CLIP 文本编码器: %s", model_name)
            self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name).to(self.device)
            self._model.eval()
            self._backend = "clip"
            self._raw_dim = 512
            return
        except Exception as exc:
            logger.warning("CLIP 加载失败 (%s)，尝试 sentence-transformers", exc)

        # --- 方案 2: Sentence-Transformers ---
        try:
            from sentence_transformers import SentenceTransformer

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info("加载 SentenceTransformer: %s", model_name)
            self._model = SentenceTransformer(model_name, device=self.device)
            self._backend = "sbert"
            self._raw_dim = 384
            return
        except ImportError as exc:
            raise ImportError(
                "意图向量编码需要以下依赖之一:\n"
                "  pip install transformers\n"
                "  pip install sentence-transformers\n"
                "若无法访问 HuggingFace，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com"
            ) from exc

    def encode(self, text: str) -> np.ndarray:
        """
        编码单条意图文本。

        Args:
            text: VLM 返回的意图描述（或 JSON 序列化字符串）

        Returns:
            shape (out_dim,) 的 float32 向量，已 L2 归一化
        """
        text = (text or "").strip()
        if not text:
            return np.zeros(self.out_dim, dtype=np.float32)

        if self._backend == "clip":
            import torch

            tokens = self._tokenizer(
                text=[text], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                feat = self._model.get_text_features(**tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            vec = feat.cpu().numpy()[0].astype(np.float32)
        else:
            vec = self._model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)

        vec = _project_to_dim(vec, self.out_dim)
        return _l2_normalize(vec)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码，返回 (len(texts), out_dim)。"""
        return np.stack([self.encode(t) for t in texts], axis=0)


def build_fallback_intent_text(intent_label: str = "unknown") -> str:
    """
    VLM 调用失败时的占位文本，避免样本缺少 intent_feature 键。
    """
    return f'{{"intent":"{intent_label}","confidence":0.0,"note":"vlm_fallback"}}'


def zero_intent_vector(dim: int = 512) -> np.ndarray:
    """全零向量；CoFE 在 intent 缺失时会用零向量，但离线写入零向量便于调试。"""
    return np.zeros(dim, dtype=np.float32)


class HashIntentEncoder:
    """
    确定性哈希嵌入（仅用于 --dry_run 测试数据管线，无语义意义）。

    不依赖 HuggingFace 网络，CI / 无 GPU 环境可快速验证写盘逻辑。
    """

    def __init__(self, out_dim: int = 512, device: str = "cpu"):
        self.out_dim = out_dim
        self.device = device

    def encode(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.out_dim).astype(np.float32)
        return _l2_normalize(vec)


def cache_key_for_image(image_path: str, prompt_version: str = "v1") -> str:
    """根据图像路径与 prompt 版本生成缓存键（避免重复调用 VLM）。"""
    h = hashlib.md5()
    h.update(image_path.encode("utf-8"))
    h.update(prompt_version.encode("utf-8"))
    return h.hexdigest()
