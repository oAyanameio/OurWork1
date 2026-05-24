#!/usr/bin/env python3
"""
离线提取 VLM 语义意图特征，并写入 T2FPV 预处理 .pt 文件

================================================================================
研究框架对应关系
================================================================================
    框架要求: 使用 VLM 离线提取行人意图（左右转、避障等），作为 CoFE 的「语义锚点」
    本脚本流程:
        1. 读取 data/processed/t2fpv_{split}.pt 中每个场景样本
        2. 根据 scene_name / agent_id / frame_ids_hist 定位 FPV 图像
        3. 调用 Qwen2.5-VL（vLLM, OpenAI API）生成结构化意图 JSON
        4. 用 CLIP 文本编码器将 JSON 转为 intent_feature 向量（默认 512 维）
        5. 写回 .pt：intent_feature (N_agents, D), intent_text (可选, 便于人工检查)

================================================================================
使用前准备
================================================================================
    1. 完成轨迹预处理（需含 scene_name / agent_ids / frame_ids_hist）:
           python scripts/preprocess.py --data_root /path/to/FPVDataset

    2. 启动 vLLM 服务:
           python scripts/start_vllm.py

    3. 安装文本编码依赖（二选一）:
           pip install transformers pillow
           pip install sentence-transformers

================================================================================
用法示例
================================================================================
    # 处理训练集，默认写入 intent 字段
    python scripts/extract_intent.py --split train

    # 指定数据根目录与特征维度
    python scripts/extract_intent.py --split train val test \\
        --data_root /home/lbh/T2FPV-ow/FPVDataset \\
        --intent_dim 512

    # 仅处理前 50 个场景（调试）
    python scripts/extract_intent.py --split train --max_scenes 50

    # 使用缓存，跳过已提取的样本（断点续跑）
    python scripts/extract_intent.py --split train --resume

    # 不调用 VLM，仅用占位文本（测试数据管线）
    python scripts/extract_intent.py --split train --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# 将 src 加入路径，与 scripts/train.py 一致
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.intent_encoding import (  # noqa: E402
    HashIntentEncoder,
    IntentTextEncoder,
    build_fallback_intent_text,
    cache_key_for_image,
    zero_intent_vector,
)
from data.intent_vlm import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    FPVImageResolver,
    VLLMIntentClient,
    intent_json_to_embedding_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Qwen2.5-VL + 文本编码器离线提取 intent_feature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- 数据路径 ---
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="FPVDataset 根目录（含 imgs/、gt_dets/）。默认自动探测",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="预处理 .pt 目录，默认 OurWork1/data/processed",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="要处理的数据划分，可多个",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help='输出文件名后缀，如 "_intent" → t2fpv_train_intent.pt；默认覆盖原文件前备份',
    )

    # --- VLM 服务 ---
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="vLLM OpenAI API 根地址（与 start_vllm.py 中 port 一致）",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="qwen2.5-vl-7b",
        help="served-model-name，与 start_vllm.py 中 --served-model-name 一致",
    )
    parser.add_argument(
        "--vlm_timeout",
        type=int,
        default=120,
        help="单次 VLM 请求超时（秒）",
    )

    # --- 特征维度 ---
    parser.add_argument(
        "--intent_dim",
        type=int,
        default=512,
        help="intent_feature 向量维度，需与 config/default.yml 中 intent_feature_dim 一致",
    )
    parser.add_argument(
        "--text_encoder_device",
        type=str,
        default="cpu",
        help="CLIP / SentenceTransformer 运行设备（cpu 即可，批量不大）",
    )

    # --- 运行控制 ---
    parser.add_argument(
        "--hist_frame",
        type=str,
        default="last",
        choices=["last", "first", "middle"],
        help="使用历史窗口的哪一帧图像询问 VLM（默认最后一帧）",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="最多处理多少个场景（调试用）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若样本已有 intent_feature 则跳过",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="不调用 VLM，使用占位意图（测试写盘逻辑）",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="覆盖原 .pt 时不创建 .bak 备份",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="VLM 原始回复 JSON 缓存目录（断点续跑、避免重复计费）",
    )

    return parser.parse_args()


def resolve_data_root(explicit: Optional[str]) -> str:
    """自动探测 FPVDataset 路径。"""
    if explicit:
        return os.path.abspath(explicit)

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "T2FPV"),
        os.path.join(os.path.dirname(__file__), "..", "FPVDataset"),
        "/home/lbh/T2FPV-ow/FPVDataset",
        os.path.join(os.path.dirname(__file__), "..", "..", "T2FPV-ow", "FPVDataset"),
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.isdir(os.path.join(abs_c, "imgs")):
            logger.info("自动探测 data_root: %s", abs_c)
            return abs_c

    raise FileNotFoundError(
        "未找到 FPVDataset，请使用 --data_root 指定（需包含 imgs/ 与 gt_dets/）"
    )


def resolve_processed_dir(explicit: Optional[str]) -> str:
    if explicit:
        return os.path.abspath(explicit)
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    )


def pick_frame_id(frame_ids_hist: np.ndarray, strategy: str) -> int:
    """从历史帧 ID 列表中选取用于 VLM 的一帧。"""
    if frame_ids_hist is None or len(frame_ids_hist) == 0:
        raise ValueError("frame_ids_hist 为空，无法定位图像")

    if strategy == "first":
        return int(frame_ids_hist[0])
    if strategy == "middle":
        return int(frame_ids_hist[len(frame_ids_hist) // 2])
    return int(frame_ids_hist[-1])


def load_samples(pt_path: str) -> List[Dict[str, Any]]:
    """加载预处理 .pt（支持 list 或 dict 打包格式）。"""
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "pos" in raw:
        n = len(raw["pos"])
        samples = []
        for i in range(n):
            samples.append({k: raw[k][i] for k in raw})
        return samples
    raise ValueError(f"无法解析 {pt_path} 的数据格式")


def save_samples(pt_path: str, samples: List[Dict[str, Any]]) -> None:
    torch.save(samples, pt_path)


def ensure_scene_metadata(scene: Dict[str, Any], scene_idx: int) -> bool:
    """
    检查样本是否包含图像定位所需的元数据。

    若缺少 scene_name / agent_ids，需重新运行 preprocess.py（已更新写入逻辑）。
    """
    if "scene_name" not in scene or "agent_ids" not in scene:
        logger.error(
            "场景 %d 缺少 scene_name 或 agent_ids。"
            "请重新运行: python scripts/preprocess.py --data_root <FPVDataset>",
            scene_idx,
        )
        return False
    if "frame_ids_hist" not in scene:
        logger.error("场景 %d 缺少 frame_ids_hist", scene_idx)
        return False
    return True


def load_vlm_cache(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    path = cache_dir / f"{key}.json"
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_vlm_cache(cache_dir: Path, key: str, payload: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_split(
    split: str,
    args: argparse.Namespace,
    data_root: str,
    processed_dir: str,
    image_resolver: FPVImageResolver,
    vlm_client: Optional[VLLMIntentClient],
    text_encoder: IntentTextEncoder,
) -> None:
    """处理单个数据划分（train / val / test）。"""
    in_name = f"t2fpv_{split}.pt"
    in_path = os.path.join(processed_dir, in_name)
    if not os.path.isfile(in_path):
        logger.warning("跳过 %s：文件不存在 %s", split, in_path)
        return

    out_path = os.path.join(
        processed_dir,
        f"t2fpv_{split}{args.output_suffix}.pt" if args.output_suffix else in_name,
    )

    # 覆盖原文件时先备份
    if not args.output_suffix and not args.no_backup and os.path.isfile(in_path):
        bak_path = in_path + ".bak"
        if not os.path.isfile(bak_path):
            import shutil

            shutil.copy2(in_path, bak_path)
            logger.info("已备份: %s", bak_path)

    samples = load_samples(in_path)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(processed_dir) / "intent_vlm_cache" / split

    n_scenes = len(samples) if args.max_scenes is None else min(len(samples), args.max_scenes)
    n_agents_total = 0
    n_vlm_calls = 0
    n_skip_resume = 0
    n_missing_image = 0

    for scene_idx in range(n_scenes):
        scene = samples[scene_idx]
        if not ensure_scene_metadata(scene, scene_idx):
            continue

        scene_name = scene["scene_name"]
        agent_ids = list(scene["agent_ids"])
        n_agents = len(agent_ids)
        frame_ids_hist = scene["frame_ids_hist"]
        if isinstance(frame_ids_hist, torch.Tensor):
            frame_ids_hist = frame_ids_hist.numpy()

        try:
            query_frame_id = pick_frame_id(frame_ids_hist, args.hist_frame)
        except ValueError as exc:
            logger.warning("场景 %d: %s", scene_idx, exc)
            continue

        intent_features = []
        intent_texts = []

        for agent_i, agent_id in enumerate(agent_ids):
            n_agents_total += 1

            # --- resume: 已有向量则跳过 ---
            if args.resume and "intent_feature" in scene:
                existing = scene["intent_feature"]
                if isinstance(existing, torch.Tensor):
                    existing = existing.numpy()
                if existing.shape[0] > agent_i and np.any(existing[agent_i] != 0):
                    n_skip_resume += 1
                    intent_features.append(existing[agent_i].astype(np.float32))
                    if "intent_text" in scene:
                        intent_texts.append(scene["intent_text"][agent_i])
                    else:
                        intent_texts.append("")
                    continue

            image_path = image_resolver.resolve(scene_name, int(agent_id), query_frame_id)
            if image_path is None:
                n_missing_image += 1
                logger.debug(
                    "无图像 scene=%s agent=%s frame=%s",
                    scene_name,
                    agent_id,
                    query_frame_id,
                )
                intent_features.append(zero_intent_vector(args.intent_dim))
                intent_texts.append(build_fallback_intent_text("no_image"))
                continue

            cache_key = cache_key_for_image(image_path, DEFAULT_PROMPT_VERSION)
            cached = load_vlm_cache(cache_dir, cache_key)

            if cached is not None:
                intent_obj = cached.get("parsed", {})
                embed_text = cached.get("embed_text") or intent_json_to_embedding_text(intent_obj)
            elif args.dry_run:
                intent_obj = {
                    "intent": "不确定",
                    "description": "dry_run",
                    "confidence": 0.0,
                }
                embed_text = intent_json_to_embedding_text(intent_obj)
            else:
                assert vlm_client is not None
                intent_obj, raw_reply = vlm_client.infer_intent_from_image(image_path)
                embed_text = intent_json_to_embedding_text(intent_obj)
                save_vlm_cache(
                    cache_dir,
                    cache_key,
                    {
                        "image_path": image_path,
                        "parsed": intent_obj,
                        "raw": raw_reply,
                        "embed_text": embed_text,
                    },
                )
                n_vlm_calls += 1

            vec = text_encoder.encode(embed_text)
            intent_features.append(vec)
            intent_texts.append(embed_text)

        # 写入场景级字段: (N_agents, intent_dim)
        scene["intent_feature"] = torch.from_numpy(np.stack(intent_features, axis=0))
        scene["intent_text"] = intent_texts

        if (scene_idx + 1) % 10 == 0 or scene_idx == n_scenes - 1:
            logger.info(
                "  [%s] 进度 %d/%d | VLM 调用 %d | 跳过(resume) %d | 缺图 %d",
                split,
                scene_idx + 1,
                n_scenes,
                n_vlm_calls,
                n_skip_resume,
                n_missing_image,
            )

    save_samples(out_path, samples)
    logger.info(
        "[%s] 完成: 保存至 %s | 场景=%d | agent=%d | VLM调用=%d",
        split,
        out_path,
        n_scenes,
        n_agents_total,
        n_vlm_calls,
    )


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    processed_dir = resolve_processed_dir(args.processed_dir)

    logger.info("data_root      = %s", data_root)
    logger.info("processed_dir  = %s", processed_dir)
    logger.info("intent_dim     = %d", args.intent_dim)
    logger.info("splits         = %s", args.split)

    image_resolver = FPVImageResolver(data_root=data_root)

    vlm_client: Optional[VLLMIntentClient] = None
    if not args.dry_run:
        vlm_client = VLLMIntentClient(
            base_url=args.vllm_base_url,
            model_name=args.vllm_model,
            timeout_sec=args.vlm_timeout,
        )
        if not vlm_client.is_server_alive():
            raise ConnectionError(
                "Qwen2.5-VL vLLM 服务未启动。\n"
                "请先运行: python scripts/start_vllm.py\n"
                "然后检查: curl http://127.0.0.1:8000/v1/models"
            )
        logger.info("VLM 服务可用: %s", args.vllm_base_url)
    else:
        logger.warning("dry_run 模式：不调用 VLM，使用占位意图")

    if args.dry_run:
        logger.info("dry_run: 使用 HashIntentEncoder（仅测试管线，无语义）")
        text_encoder = HashIntentEncoder(out_dim=args.intent_dim)
    else:
        logger.info("加载文本编码器（CLIP 或 SentenceTransformer）...")
        text_encoder = IntentTextEncoder(
            out_dim=args.intent_dim, device=args.text_encoder_device
        )

    t0 = time.time()
    for split in args.split:
        process_split(
            split,
            args,
            data_root,
            processed_dir,
            image_resolver,
            vlm_client,
            text_encoder,
        )

    logger.info("全部完成，耗时 %.1f s", time.time() - t0)
    logger.info(
        "下一步: 在 config/default.yml 设置 intent_feature_dim: %d，然后训练 CoFE / PTINet",
        args.intent_dim,
    )


if __name__ == "__main__":
    main()
