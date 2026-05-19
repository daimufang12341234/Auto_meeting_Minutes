"""阿里云百炼 语音转文字 SDK 调用脚本

三种方式:
  1. QwenASR      - 千问语音识别 (OpenAI兼容 / DashScope同步)
  2. FunASR       - Fun-ASR录音文件识别 (Transcription SDK 异步)
  3. ParaformerASR - Paraformer录音文件识别 (Transcription SDK 异步)

安装依赖: uv pip install openai dashscope
配置文件: config.ini (使用 ini_config 包加载)
"""

import os
import time
import base64
import json
import pathlib
from typing import Optional

from ini_config import IniConfig


# ─── 配置加载 ───────────────────────────────────────────

def load_config(path: str = "config.ini") -> IniConfig:
    """加载 INI 配置文件，优先从脚本所在目录查找"""
    script_dir = pathlib.Path(__file__).resolve().parent
    candidate = script_dir / path
    if candidate.exists():
        return IniConfig(str(candidate))
    return IniConfig(path)


_DEFAULT_CONFIG = None


def _get_config(path: Optional[str] = None) -> IniConfig:
    global _DEFAULT_CONFIG
    if path:
        return IniConfig(path)
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load_config()
    return _DEFAULT_CONFIG


def _bool(value, default=False):
    """将 INI 字符串值转为 bool"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


# ─── 工具函数 ───────────────────────────────────────────

def _api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或在 config.ini 中配置 api_key")
    return key


def _encode_audio_base64(file_path: str) -> str:
    """本地音频 → Base64 data URI"""
    mime_map = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma", ".webm": "audio/webm",
    }
    ext = pathlib.Path(file_path).suffix.lower()
    mime = mime_map.get(ext, "audio/mpeg")
    data = pathlib.Path(file_path).read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _resolve_api_key(explicit: Optional[str], config: IniConfig) -> str:
    """解析 API Key: 显式参数 > config > 环境变量"""
    key = explicit or config.api.api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 或在 config.ini [api] 节中配置 api_key")
    return key


# ─── 方式1: Qwen-ASR ───────────────────────────────────

class QwenASR:
    """千问语音识别 (OpenAI兼容 / DashScope同步), 同步调用, 适合短音频 (≤10MB)

    用法:
        asr = QwenASR()                                    # 使用 config.ini 默认配置
        asr = QwenASR(api_key="sk-xxx", region="intl")     # 显式覆盖
        asr = QwenASR(config="my_config.ini")              # 指定配置文件
        result = asr.transcribe("audio.wav")
    """

    OPENAI_URL_CN = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_URL_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_URL_CN = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    DASHSCOPE_URL_INTL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def __init__(self, api_key: Optional[str] = None, region: Optional[str] = None,
                 config: Optional[str] = None,
                 model: Optional[str] = None,
                 language: Optional[str] = None,
                 enable_itn: Optional[bool] = None):
        cfg = _get_config(config)

        self.api_key = _resolve_api_key(api_key, cfg)
        self.region = region or cfg.api.region or "cn"

        if self.region == "intl":
            self.openai_url = self.OPENAI_URL_INTL
            self.dashscope_url = self.DASHSCOPE_URL_INTL
        else:
            self.openai_url = self.OPENAI_URL_CN
            self.dashscope_url = self.DASHSCOPE_URL_CN

        self.model = model or cfg.qwen.model or "qwen3-asr-flash"
        self.language = language or cfg.qwen.language or None
        self.enable_itn = enable_itn if enable_itn is not None else _bool(cfg.qwen.enable_itn)

    # ── OpenAI 兼容模式 ──

    def transcribe_openai(self, input_data: str, is_url: bool = False,
                          language: Optional[str] = None,
                          enable_itn: Optional[bool] = None) -> dict:
        """OpenAI 兼容接口: input_data 为本地文件路径或公网 URL 或 Base64 data URI"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.openai_url)

        if not is_url and not input_data.startswith("data:"):
            audio = _encode_audio_base64(input_data)
        else:
            audio = input_data

        _itn = enable_itn if enable_itn is not None else self.enable_itn
        extra_body = {"asr_options": {"enable_itn": _itn}}
        lang = language or self.language
        if lang:
            extra_body["asr_options"]["language"] = lang

        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio}}]}],
            extra_body=extra_body,
        )
        return {
            "text": completion.choices[0].message.content,
            "model": completion.model,
            "usage": {"total_tokens": completion.usage.total_tokens},
        }

    # ── DashScope 同步模式 ──

    def transcribe_dashscope(self, input_data: str,
                             language: Optional[str] = None,
                             enable_itn: Optional[bool] = None) -> dict:
        """DashScope 同步接口: 支持本地文件路径、公网 URL、Base64 data URI"""
        import requests

        _itn = enable_itn if enable_itn is not None else self.enable_itn
        asr_options = {"enable_itn": _itn}
        lang = language or self.language
        if lang:
            asr_options["language"] = lang

        if input_data.startswith("http://") or input_data.startswith("https://"):
            audio_field = input_data
        elif input_data.startswith("data:"):
            audio_field = input_data
        else:
            audio_field = "file://" + os.path.abspath(input_data)

        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"audio": audio_field}]}]},
            "parameters": {"asr_options": asr_options},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.dashscope_url, headers=headers, json=payload, timeout=120)
        data = resp.json()

        output = data.get("output", {})
        choices = output.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", [])
            text = "".join(c.get("text", "") for c in content)
            return {"text": text, "usage": data.get("usage", {})}
        return data

    # ── 便捷方法 ──

    def transcribe(self, input_data: str, **kwargs) -> dict:
        """自动选择: 本地文件用 OpenAI 兼容 + Base64, URL 用 DashScope"""
        if input_data.startswith("http://") or input_data.startswith("https://"):
            return self.transcribe_dashscope(input_data, **kwargs)
        else:
            return self.transcribe_openai(input_data, **kwargs)


# ─── 方式2: Fun-ASR 录音文件识别 ──────────────────────────

class FunASR:
    """Fun-ASR 录音文件识别 (Transcription SDK 异步调用)
    适合长音频 (≤2GB, ≤12h), 支持说话人分离

    用法:
        asr = FunASR()                                    # 使用 config.ini 默认配置
        asr = FunASR(api_key="sk-xxx", region="intl")     # 显式覆盖
        asr = FunASR(config="my_config.ini")              # 指定配置文件
        result = asr.transcribe(["https://oss.example.com/audio.wav"])
        task_id = asr.submit(["url1", "url2"])
        result = asr.fetch(task_id)
    """

    SUPPORTED_MODELS = [
        "fun-asr",
        "fun-asr-mtl",
        "fun-asr-2025-11-07",
        "fun-asr-mtl-2025-08-25",
        "fun-asr-2025-08-25",
    ]

    def __init__(self, api_key: Optional[str] = None, region: Optional[str] = None,
                 config: Optional[str] = None,
                 model: Optional[str] = None,
                 diarization_enabled: Optional[bool] = None,
                 speaker_count: Optional[int] = None,
                 language: Optional[str] = None,
                 poll_interval: Optional[float] = None,
                 timeout: Optional[float] = None):
        cfg = _get_config(config)

        self.api_key = _resolve_api_key(api_key, cfg)
        self.region = region or cfg.api.region or "cn"

        import dashscope
        if self.region == "intl":
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        else:
            dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        dashscope.api_key = self.api_key
        self._base_url = dashscope.base_http_api_url

        self.model = model or cfg.fun_asr.model or "fun-asr"
        self.diarization_enabled = diarization_enabled if diarization_enabled is not None else _bool(cfg.fun_asr.diarization_enabled)
        self.speaker_count = speaker_count if speaker_count is not None else cfg.get_int("fun_asr", "speaker_count", 0)
        self.language = language or cfg.fun_asr.language or None
        self.poll_interval = poll_interval or float(cfg.fun_asr.poll_interval or 3)
        self.timeout = timeout or float(cfg.fun_asr.timeout or 3600)

    def transcribe(self, file_urls: list, model: Optional[str] = None,
                   diarization_enabled: Optional[bool] = None,
                   speaker_count: Optional[int] = None,
                   language: Optional[str] = None,
                   ) -> dict:
        """同步等待: 提交任务后阻塞直到完成"""
        from http import HTTPStatus
        from dashscope.audio.asr import Transcription

        kwargs = {
            "model": model or self.model,
            "file_urls": file_urls,
        }
        _dia = diarization_enabled if diarization_enabled is not None else self.diarization_enabled
        if _dia:
            kwargs["diarization_enabled"] = True
        _spk = speaker_count if speaker_count is not None else self.speaker_count
        if _spk:
            kwargs["speaker_count"] = _spk
        lang = language or self.language
        if lang:
            kwargs["language_hints"] = [lang]

        task_resp = Transcription.async_call(**kwargs)
        task_id = task_resp.output.task_id
        print(f"[FunASR] 任务已提交: {task_id}")

        transcribe_resp = Transcription.wait(task=task_id)
        if transcribe_resp.status_code == HTTPStatus.OK:
            return self._parse_result(transcribe_resp)
        return {"task_id": task_id, "error": "任务失败或超时"}

    def submit(self, file_urls: list, model: Optional[str] = None,
               diarization_enabled: Optional[bool] = None,
               speaker_count: Optional[int] = None,
               language: Optional[str] = None) -> str:
        """仅提交任务, 返回 task_id"""
        from dashscope.audio.asr import Transcription

        kwargs = {
            "model": model or self.model,
            "file_urls": file_urls,
        }
        _dia = diarization_enabled if diarization_enabled is not None else self.diarization_enabled
        if _dia:
            kwargs["diarization_enabled"] = True
        _spk = speaker_count if speaker_count is not None else self.speaker_count
        if _spk:
            kwargs["speaker_count"] = _spk
        lang = language or self.language
        if lang:
            kwargs["language_hints"] = [lang]

        task_resp = Transcription.async_call(**kwargs)
        return task_resp.output.task_id

    def fetch(self, task_id: str) -> dict:
        """查询任务结果"""
        from http import HTTPStatus
        from dashscope.audio.asr import Transcription

        resp = Transcription.fetch(task=task_id)
        if resp.status_code == HTTPStatus.OK:
            output = resp.output
            if output.task_status == "SUCCEEDED":
                return self._parse_result(resp)
            return {"task_id": task_id, "status": output.task_status}
        return {"task_id": task_id, "error": "查询失败"}

    def fetch_wait(self, task_id: str, poll_interval: Optional[float] = None,
                   timeout: Optional[float] = None) -> dict:
        """轮询等待任务完成并返回结果"""
        import requests

        _poll = poll_interval or self.poll_interval
        _timeout = timeout or self.timeout

        query_url = f"{self._base_url}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
        }
        start = time.time()

        while True:
            time.sleep(_poll)
            resp = requests.get(query_url, headers=headers, timeout=30)
            data = resp.json()
            status = data.get("output", {}).get("task_status")

            if status == "SUCCEEDED":
                results = data["output"].get("results", [])
                transcripts = []
                for item in results:
                    url = item.get("transcription_url")
                    if url:
                        try:
                            r = requests.get(url, timeout=30)
                            transcripts.append(r.json())
                        except Exception as e:
                            transcripts.append({"error": str(e)})
                return {"task_id": task_id, "status": "SUCCEEDED", "transcripts": transcripts}

            if status == "FAILED":
                return {"task_id": task_id, "status": "FAILED", "output": data["output"]}

            if time.time() - start > _timeout:
                return {"task_id": task_id, "status": "TIMEOUT"}

            print(f"  状态: {status}, 已等待 {time.time() - start:.0f}s")

    def _parse_result(self, resp) -> dict:
        """解析 TranscriptionResponse, 下载识别结果"""
        import requests

        output = resp.output
        results = output.get("results") if hasattr(output, "results") else output.results
        if results is None:
            results = []

        transcripts = []
        for item in results:
            file_url = item.get("file_url") if isinstance(item, dict) else getattr(item, "file_url", None)
            transcription_url = item.get("transcription_url") if isinstance(item, dict) else getattr(item, "transcription_url", None)
            subtask_status = item.get("subtask_status") if isinstance(item, dict) else getattr(item, "subtask_status", None)

            if transcription_url:
                try:
                    r = requests.get(transcription_url, timeout=30)
                    transcript = r.json()
                    transcript["_file_url"] = file_url
                    transcript["_subtask_status"] = subtask_status
                    transcripts.append(transcript)
                except Exception as e:
                    transcripts.append({"error": str(e), "file_url": file_url})

        return {
            "task_id": output.task_id,
            "task_status": output.task_status,
            "usage": getattr(resp, "usage", None),
            "transcripts": transcripts,
        }


# ─── 方式3: Paraformer 录音文件识别 ─────────────────────────

class ParaformerASR:
    """Paraformer 录音文件识别 (Transcription SDK 异步调用)
    适合长音频 (≤2GB, ≤12h), 支持说话人分离、语气词过滤、时间戳校准

    用法:
        asr = ParaformerASR()                             # 使用 config.ini 默认配置
        asr = ParaformerASR(api_key="sk-xxx", region="intl")  # 显式覆盖
        asr = ParaformerASR(config="my_config.ini")       # 指定配置文件
        result = asr.transcribe(["https://oss.example.com/audio.wav"])
        task_id = asr.submit(["url1", "url2"])
        result = asr.fetch(task_id)
    """

    SUPPORTED_MODELS = [
        "paraformer-v2",
        "paraformer-v1",
    ]

    def __init__(self, api_key: Optional[str] = None, region: Optional[str] = None,
                 config: Optional[str] = None,
                 model: Optional[str] = None,
                 diarization_enabled: Optional[bool] = None,
                 speaker_count: Optional[int] = None,
                 language_hints: Optional[list] = None,
                 disfluency_removal_enabled: Optional[bool] = None,
                 timestamp_alignment_enabled: Optional[bool] = None):
        cfg = _get_config(config)

        self.api_key = _resolve_api_key(api_key, cfg)
        self.region = region or cfg.api.region or "cn"

        import dashscope
        if self.region == "intl":
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        else:
            dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        dashscope.api_key = self.api_key
        self._base_url = dashscope.base_http_api_url

        self.model = model or cfg.paraformer.model or "paraformer-v2"
        self.diarization_enabled = diarization_enabled if diarization_enabled is not None else _bool(cfg.paraformer.diarization_enabled)
        self.speaker_count = speaker_count if speaker_count is not None else cfg.get_int("paraformer", "speaker_count", 0)
        self.disfluency_removal_enabled = disfluency_removal_enabled if disfluency_removal_enabled is not None else _bool(cfg.paraformer.disfluency_removal_enabled)
        self.timestamp_alignment_enabled = timestamp_alignment_enabled if timestamp_alignment_enabled is not None else _bool(cfg.paraformer.timestamp_alignment_enabled)

        # 解析 language_hints: 显式 > config
        if language_hints is not None:
            self.language_hints = language_hints
        else:
            raw = cfg.paraformer.language_hints
            if raw:
                self.language_hints = [x.strip() for x in raw.split(",") if x.strip()]
            else:
                self.language_hints = None

    def transcribe(self, file_urls: list, model: Optional[str] = None,
                   diarization_enabled: Optional[bool] = None,
                   speaker_count: Optional[int] = None,
                   language_hints: Optional[list] = None,
                   disfluency_removal_enabled: Optional[bool] = None,
                   timestamp_alignment_enabled: Optional[bool] = None,
                   channel_id: list = None) -> dict:
        """同步等待: 提交任务后阻塞直到完成"""
        from http import HTTPStatus
        from dashscope.audio.asr import Transcription

        kwargs = {
            "model": model or self.model,
            "file_urls": file_urls,
        }
        _dia = diarization_enabled if diarization_enabled is not None else self.diarization_enabled
        if _dia:
            kwargs["diarization_enabled"] = True
        _spk = speaker_count if speaker_count is not None else self.speaker_count
        if _spk:
            kwargs["speaker_count"] = _spk
        _lang = language_hints if language_hints is not None else self.language_hints
        if _lang:
            kwargs["language_hints"] = _lang
        _dis = disfluency_removal_enabled if disfluency_removal_enabled is not None else self.disfluency_removal_enabled
        if _dis:
            kwargs["disfluency_removal_enabled"] = True
        _ts = timestamp_alignment_enabled if timestamp_alignment_enabled is not None else self.timestamp_alignment_enabled
        if _ts:
            kwargs["timestamp_alignment_enabled"] = True
        if channel_id is not None:
            kwargs["channel_id"] = channel_id

        task_resp = Transcription.async_call(**kwargs)
        task_id = task_resp.output.task_id
        print(f"[ParaformerASR] 任务已提交: {task_id}")

        transcribe_resp = Transcription.wait(task=task_id)
        if transcribe_resp.status_code == HTTPStatus.OK:
            return self._parse_result(transcribe_resp)
        return {"task_id": task_id, "error": "任务失败或超时"}

    def submit(self, file_urls: list, model: Optional[str] = None,
               diarization_enabled: Optional[bool] = None,
               speaker_count: Optional[int] = None,
               language_hints: Optional[list] = None,
               disfluency_removal_enabled: Optional[bool] = None,
               timestamp_alignment_enabled: Optional[bool] = None,
               channel_id: list = None) -> str:
        """仅提交任务, 返回 task_id"""
        from dashscope.audio.asr import Transcription

        kwargs = {
            "model": model or self.model,
            "file_urls": file_urls,
        }
        _dia = diarization_enabled if diarization_enabled is not None else self.diarization_enabled
        if _dia:
            kwargs["diarization_enabled"] = True
        _spk = speaker_count if speaker_count is not None else self.speaker_count
        if _spk:
            kwargs["speaker_count"] = _spk
        _lang = language_hints if language_hints is not None else self.language_hints
        if _lang:
            kwargs["language_hints"] = _lang
        _dis = disfluency_removal_enabled if disfluency_removal_enabled is not None else self.disfluency_removal_enabled
        if _dis:
            kwargs["disfluency_removal_enabled"] = True
        _ts = timestamp_alignment_enabled if timestamp_alignment_enabled is not None else self.timestamp_alignment_enabled
        if _ts:
            kwargs["timestamp_alignment_enabled"] = True
        if channel_id is not None:
            kwargs["channel_id"] = channel_id

        task_resp = Transcription.async_call(**kwargs)
        return task_resp.output.task_id

    def fetch(self, task_id: str) -> dict:
        """查询任务结果"""
        from http import HTTPStatus
        from dashscope.audio.asr import Transcription

        resp = Transcription.fetch(task=task_id)
        if resp.status_code == HTTPStatus.OK:
            output = resp.output
            if output.task_status == "SUCCEEDED":
                return self._parse_result(resp)
            return {"task_id": task_id, "status": output.task_status}
        return {"task_id": task_id, "error": "查询失败"}

    def _parse_result(self, resp) -> dict:
        """解析 TranscriptionResponse, 下载识别结果"""
        import requests

        output = resp.output
        results = output.get("results") if hasattr(output, "results") else output.results
        if results is None:
            results = []

        transcripts = []
        for item in results:
            file_url = item.get("file_url") if isinstance(item, dict) else getattr(item, "file_url", None)
            transcription_url = item.get("transcription_url") if isinstance(item, dict) else getattr(item, "transcription_url", None)
            subtask_status = item.get("subtask_status") if isinstance(item, dict) else getattr(item, "subtask_status", None)

            if transcription_url:
                try:
                    r = requests.get(transcription_url, timeout=30)
                    transcript = r.json()
                    transcript["_file_url"] = file_url
                    transcript["_subtask_status"] = subtask_status
                    transcripts.append(transcript)
                except Exception as e:
                    transcripts.append({"error": str(e), "file_url": file_url})

        return {
            "task_id": output.task_id,
            "task_status": output.task_status,
            "usage": getattr(resp, "usage", None),
            "transcripts": transcripts,
        }


# ─── 命令行入口 ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="阿里云百炼语音转文字")
    parser.add_argument("input", help="音频文件路径 或 公网 URL")
    parser.add_argument("-t", "--type", choices=["qwen", "fun", "paraformer"],
                        default=None, help="调用方式: qwen / fun / paraformer (默认从 config.ini 读取)")
    parser.add_argument("--model", default=None, help="模型名称 (覆盖配置文件)")
    parser.add_argument("--diarization", action="store_true", default=None, help="启用说话人分离 (fun/paraformer)")
    parser.add_argument("--speakers", type=int, default=None, help="说话人数量 (fun/paraformer)")
    parser.add_argument("--language", default=None, help="语言代码, e.g. zh, en")
    parser.add_argument("--region", choices=["cn", "intl"], default=None, help="地域")
    parser.add_argument("--output", "-o", help="结果保存路径")
    parser.add_argument("-c", "--config", default="config.ini", help="配置文件路径 (默认: config.ini)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 如果命令行未指定 -t, 从配置文件 [asr] method 读取
    asr_type = args.type or cfg.asr.method or "qwen"

    if asr_type == "qwen":
        asr = QwenASR(
            config=args.config,
            region=args.region,
            model=args.model,
            language=args.language,
        )
        result = asr.transcribe(args.input)

    elif asr_type == "fun":
        asr = FunASR(
            config=args.config,
            region=args.region,
            model=args.model,
            diarization_enabled=args.diarization,
            speaker_count=args.speakers,
            language=args.language,
        )
        if not args.input.startswith("http"):
            print("Fun-ASR 需要公网可访问的音频 URL")
            exit(1)
        result = asr.transcribe([args.input])

    elif asr_type == "paraformer":
        asr = ParaformerASR(
            config=args.config,
            region=args.region,
            model=args.model,
            diarization_enabled=args.diarization,
            speaker_count=args.speakers,
            language_hints=[args.language] if args.language else None,
        )
        if not args.input.startswith("http"):
            print("Paraformer 需要公网可访问的音频 URL")
            exit(1)
        result = asr.transcribe([args.input])

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
