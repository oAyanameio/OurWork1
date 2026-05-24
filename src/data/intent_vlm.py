"""
Qwen2.5-VL（vLLM OpenAI 兼容 API）调用与 FPV 图像路径解析

图像路径约定（T2FPV / FPVDataset）:
    {data_root}/imgs/{scene_name}/agent{id}_seg/idx{k}.jpg

其中 k 为该 agent 检测 CSV 中 frame_id 对应的行序号（从 0 起），
由本模块在首次访问时缓存 frame_id → idx 映射。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLM Prompt：要求结构化 JSON，便于解析与后续文本编码
# ---------------------------------------------------------------------------
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_SYSTEM_PROMPT = (
    "你是第一人称视角（FPV）下的行人行为分析助手。"
    "根据图像中正在运动或即将影响自车路径的行人，判断其短期运动意图。"
)
DEFAULT_USER_PROMPT = """请分析图像中【主要行人】的短期运动意图（约未来 1–2 秒），从下列标签中选最贴切的一项：
直行、左转、右转、减速、加速、避让、横穿、停留、不确定

只输出一行 JSON，不要其它文字：
{"intent":"<标签>","description":"<一句中文描述>","confidence":<0到1的小数>}"""


def parse_vlm_json_response(raw: str) -> Dict[str, Any]:
    """
    从 VLM 回复中解析 JSON。

    兼容模型在 JSON 外包裹 markdown 代码块的情况。
    """
    raw = (raw or "").strip()
    if not raw:
        return {"intent": "不确定", "description": "", "confidence": 0.0}

    # 提取 ```json ... ``` 或首个 { ... } 块
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if code_match:
        raw = code_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            raw = brace_match.group(0)

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 解析失败：整段文本作为 description
    return {
        "intent": "不确定",
        "description": raw[:200],
        "confidence": 0.0,
        "parse_error": True,
    }


def intent_json_to_embedding_text(obj: Dict[str, Any]) -> str:
    """
    将 JSON 意图转为供 CLIP / SentenceTransformer 编码的短文本。

    同时保留 intent 标签与 description，增强语义密度。
    """
    intent = str(obj.get("intent", "不确定"))
    desc = str(obj.get("description", ""))
    conf = obj.get("confidence", 0.0)
    return f"行人意图:{intent}。{desc}。置信度:{conf}"


class FrameIdIndexCache:
    """
    缓存每个 (scene, agent) 的 CSV，用于 frame_id → 图像 idx 映射。
    """

    def __init__(self, gt_dets_root: str):
        self.gt_dets_root = Path(gt_dets_root)
        self._csv_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    def _load_agent_csv(self, scene_name: str, agent_id: int) -> Optional[pd.DataFrame]:
        key = (scene_name, agent_id)
        if key in self._csv_cache:
            return self._csv_cache[key]

        csv_path = self.gt_dets_root / scene_name / f"agent{agent_id}_dets.csv"
        if not csv_path.is_file():
            logger.warning("检测 CSV 不存在: %s", csv_path)
            self._csv_cache[key] = None
            return None

        df = pd.read_csv(csv_path)
        if "frame_id" not in df.columns:
            self._csv_cache[key] = None
            return None

        df = df.drop_duplicates(subset="frame_id").sort_values("frame_id").reset_index(drop=True)
        self._csv_cache[key] = df
        return df

    def frame_id_to_image_idx(self, scene_name: str, agent_id: int, frame_id: int) -> Optional[int]:
        """
        将全局 frame_id 映射为 imgs/.../idx{k}.jpg 中的 k。

        k 定义为该 agent CSV 中 frame_id 排序后的行号。
        """
        df = self._load_agent_csv(scene_name, agent_id)
        if df is None:
            return None

        matches = np.where(df["frame_id"].values == frame_id)[0]
        if len(matches) == 0:
            return None
        return int(matches[0])


class FPVImageResolver:
    """根据场景名、agent_id、frame_id 解析 FPV 图像绝对路径。"""

    def __init__(self, data_root: str, gt_dets_rel: str = "gt_dets", imgs_rel: str = "imgs"):
        self.data_root = Path(data_root)
        self.imgs_root = self.data_root / imgs_rel
        self.frame_cache = FrameIdIndexCache(str(self.data_root / gt_dets_rel))

    def resolve(
        self,
        scene_name: str,
        agent_id: int,
        frame_id: int,
        prefer_seg: bool = True,
    ) -> Optional[str]:
        """
        返回图像路径；优先 agent{id}_seg 目录（与 T2FPV 裁剪行人 patch 一致）。
        """
        idx = self.frame_cache.frame_id_to_image_idx(scene_name, agent_id, frame_id)
        if idx is None:
            return None

        subdirs = [f"agent{agent_id}_seg", f"agent{agent_id}"] if prefer_seg else [f"agent{agent_id}"]
        for sub in subdirs:
            candidate = self.imgs_root / scene_name / sub / f"idx{idx}.jpg"
            if candidate.is_file():
                return str(candidate)
            # 部分数据使用 .png
            candidate_png = candidate.with_suffix(".png")
            if candidate_png.is_file():
                return str(candidate_png)

        return None


class VLLMIntentClient:
    """
    通过 vLLM OpenAI 兼容接口调用 Qwen2.5-VL-7B。

    默认连接 scripts/start_vllm.py 启动的服务:
        http://127.0.0.1:8000/v1/chat/completions
        model: qwen2.5-vl-7b
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model_name: str = "qwen2.5-vl-7b",
        timeout_sec: int = 120,
        max_tokens: int = 128,
    ):
        self.chat_url = base_url.rstrip("/") + "/chat/completions"
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens

    @staticmethod
    def _encode_image_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def is_server_alive(self) -> bool:
        """检查 vLLM 服务是否可达。"""
        import urllib.error
        import urllib.request

        models_url = self.chat_url.replace("/chat/completions", "/models")
        try:
            with urllib.request.urlopen(models_url, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def infer_intent_from_image(
        self,
        image_path: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
    ) -> Tuple[Dict[str, Any], str]:
        """
        调用 VLM，返回 (解析后的 dict, 原始回复文本)。
        """
        import urllib.error
        import urllib.request

        b64 = self._encode_image_b64(image_path)
        # Qwen2.5-VL 多模态消息格式（OpenAI 兼容 image_url）
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }

        req = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"无法连接 VLM 服务 {self.chat_url}，请先运行: python scripts/start_vllm.py\n"
                f"原始错误: {exc}"
            ) from exc

        raw = body["choices"][0]["message"]["content"]
        return parse_vlm_json_response(raw), raw
