"""QwenASR 长音频识别管道

流程: 切分 WAV → 逐段 QwenASR 识别 → 合并结果 → 保存 TXT

用法:
    python qwen_long_audio.py <音频文件路径>
    python qwen_long_audio.py <音频文件路径> -c config_qwen_split.ini
"""

import argparse
import json
import os
import pathlib
import tempfile
import time
from pathlib import Path

import soundfile as sf

from aliyun_asr import QwenASR
from ini_config import IniConfig

WAV_HEADER_BYTES = 44
SUBTYPE_TO_BYTES = {"PCM_16": 2, "PCM_24": 3, "FLOAT": 4, "PCM_32": 4, "PCM_U8": 1}


def _bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def split_wav(input_path: str, max_size_mb: float, keep_chunks: bool,
              work_dir: pathlib.Path) -> list:
    """切分 WAV 文件, 返回 [(chunk_path, start_sec, end_sec), ...]"""
    info = sf.info(input_path)
    sr = info.samplerate
    channels = info.channels
    bytes_per_sample = SUBTYPE_TO_BYTES.get(info.subtype, 2)
    bytes_per_sec = sr * channels * bytes_per_sample

    speech, _ = sf.read(input_path)
    total_samples = len(speech)
    duration_sec = total_samples / sr
    orig_file_size = os.path.getsize(input_path)

    max_data_bytes = max_size_mb * 1024 * 1024 - WAV_HEADER_BYTES
    max_duration_per_chunk = max_data_bytes / bytes_per_sec * 0.98

    print(f"音频: {duration_sec:.1f}s, {orig_file_size / 1024 / 1024:.1f}MB, "
          f"{sr}Hz, {channels}ch, {info.subtype}")

    chunk_samples = int(max_duration_per_chunk * sr)
    if chunk_samples >= total_samples:
        print("文件大小在限制内，无需切分")
        return [(input_path, 0.0, duration_sec)]

    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    print(f"切分: 每段 ≤{max_size_mb}MB (~{max_duration_per_chunk:.0f}s), 共 {num_chunks} 段")

    stem = Path(input_path).stem
    if keep_chunks:
        chunks_dir = Path(input_path).parent / f"{stem}_qwen_chunks"
    else:
        chunks_dir = pathlib.Path(tempfile.mkdtemp(prefix="qwen_chunks_"))
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for idx in range(num_chunks):
        start_sample = idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = speech[start_sample:end_sample]

        start_sec = start_sample / sr
        end_sec = end_sample / sr

        out_path = chunks_dir / f"{stem}_{idx + 1:04d}.wav"
        sf.write(str(out_path), chunk, sr)

        actual_size = os.path.getsize(out_path)
        print(f"  [{idx + 1}/{num_chunks}] {start_sec:.1f}s-{end_sec:.1f}s "
              f"({end_sec - start_sec:.1f}s, {actual_size / 1024 / 1024:.1f}MB)")

        chunks.append((str(out_path), start_sec, end_sec))

    return chunks


def cleanup_chunks(chunks: list):
    """删除所有切分文件及其目录"""
    if not chunks:
        return
    chunks_dir = Path(chunks[0][0]).parent
    for path, _, _ in chunks:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        chunks_dir.rmdir()
    except OSError:
        pass


def transcribe_chunks(chunks: list, asr: QwenASR) -> list:
    """逐段识别, 返回 [(start_sec, end_sec, text), ...]"""
    results = []
    total = len(chunks)
    for i, (path, start_sec, end_sec) in enumerate(chunks, 1):
        print(f"  识别 [{i}/{total}] {start_sec:.1f}s-{end_sec:.1f}s ...", end=" ", flush=True)
        try:
            result = asr.transcribe(path)
            text = result.get("text", "")
            results.append((start_sec, end_sec, text))
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"OK: {preview}")
        except Exception as e:
            print(f"失败: {e}")
            results.append((start_sec, end_sec, f"[识别失败] {e}"))
    return results


def format_output(results: list) -> list:
    """格式化结果"""
    lines = []
    for i, (start_sec, end_sec, text) in enumerate(results, 1):
        if text:
            lines.append(f"{i}\t[{start_sec:.3f},{end_sec:.3f}]\t{text}")
    return lines


def save_output(lines: list, output_path: pathlib.Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"\n结果已保存: {output_path} ({len(lines)} 段)")


def process(input_path: str, config_path: str = "config_qwen_split.ini"):
    # 解析为绝对路径, 确保 QwenASR 读到同一个文件
    config_abs = str(pathlib.Path(config_path).resolve())
    cfg = IniConfig(config_abs)

    max_size_mb = float(cfg.split.max_size_mb or 3.0)
    keep_chunks = _bool(cfg.split.keep_chunks)
    result_dir = cfg.output.result_dir or "output"

    print("=" * 60)
    print("QwenASR 长音频识别管道")
    print(f"配置: {config_abs}")
    print("=" * 60)

    # 切分
    print("\n[1/2] 切分音频...")
    work_dir = Path(input_path).parent
    chunks = split_wav(input_path, max_size_mb, keep_chunks, work_dir)

    # 初始化 QwenASR (只读 config_qwen_split.ini)
    print("\n[2/2] QwenASR 逐段识别...")
    asr = QwenASR(config=config_abs)
    print(f"  模型: {asr.model}, 语言: {asr.language or '自动'}")

    start_time = time.time()
    results = transcribe_chunks(chunks, asr)
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s, 平均每段: {elapsed / len(chunks):.1f}s")

    # 合并格式化
    lines = format_output(results)

    # 保存
    output_dir = pathlib.Path(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    output_path = output_dir / f"{stem}.qwen_long.txt"
    save_output(lines, output_path)

    # JSON 备份
    json_path = output_dir / f"{stem}.qwen_long.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"start": s, "end": e, "text": t} for s, e, t in results],
                  f, indent=2, ensure_ascii=False)

    # 清理
    if not keep_chunks and len(chunks) > 1:
        cleanup_chunks(chunks)

    print("=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenASR 长音频识别管道")
    parser.add_argument("input", help="输入的 WAV 文件路径")
    parser.add_argument("-c", "--config", default="config_qwen_split.ini",
                        help="配置文件路径 (默认: config_qwen_split.ini)")
    args = parser.parse_args()

    process(args.input, args.config)
