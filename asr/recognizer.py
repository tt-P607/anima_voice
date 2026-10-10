"""FunASR 本地流式识别封装。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from ..config import AnimaVoiceConfig
from .text_quality import is_obviously_invalid_asr_text


class StreamingRecognizer:
    """缓存音频、执行端点检测并调用 FunASR。"""

    def __init__(self, config: AnimaVoiceConfig) -> None:
        """初始化识别状态，不导入或加载模型。"""

        self._config = config
        self._audio_buffer: list[Any] = []
        self._pending_samples = 0
        self._total_samples = 0
        self._last_voice_sample = 0
        self._has_voice = False
        self._text = ""
        self._endpoint = False
        self._finalized = False
        self._recognizer: Any | None = None

    @classmethod
    def validate_model_files(cls, config: AnimaVoiceConfig) -> None:
        """校验显式本地模型目录。"""

        model = config.asr.model.strip()
        if cls._looks_like_local_path(model) and not Path(model).exists():
            raise FileNotFoundError(f"FunASR 模型路径不存在: asr.model={model}")

    def accept_waveform(self, samples: Any) -> None:
        """缓存一段音频并更新静音端点状态。"""

        import numpy as np

        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return
        self._audio_buffer.append(array)
        self._pending_samples += int(array.size)
        self._total_samples += int(array.size)
        self._trim_audio_buffer()
        rms = float(np.sqrt(np.mean(np.square(array))))
        asr = self._config.asr
        if rms >= asr.endpoint_rms_threshold:
            self._last_voice_sample = self._total_samples
            self._has_voice = True
            self._endpoint = False
            self._finalized = False
            return
        silence_ms = (
            (self._total_samples - self._last_voice_sample)
            * 1000
            / self._config.audio.sample_rate
        )
        self._endpoint = bool(
            self._has_voice and silence_ms >= asr.endpoint_silence_ms
        )

    def decode(self) -> None:
        """达到流式窗口后执行中间推理。"""

        asr = self._config.asr
        if not asr.enable_streaming:
            return
        window_samples = max(
            1, int(self._config.audio.sample_rate * asr.inference_window_ms / 1000)
        )
        if self._pending_samples < window_samples:
            return
        self._run_inference(is_final=False)
        self._pending_samples = 0

    def get_text(self) -> str:
        """在端点处执行最终推理并返回识别文本。"""

        if self._endpoint and not self._finalized:
            self._run_inference(is_final=True)
            self._finalized = True
        return self._text.strip()

    def get_token_confidences(self) -> list[float]:
        """返回后端可用的 token 置信度；FunASR AutoModel 不提供时为空。"""

        return []

    def get_avg_confidence(self) -> float | None:
        """返回后端可用的平均置信度；当前 FunASR 后端不提供。"""

        return None

    def is_endpoint(self) -> bool:
        """返回端点检测状态。"""

        return bool(self._config.asr.enable_endpoint and self._endpoint)

    def has_audio(self) -> bool:
        """返回是否缓存了识别音频。"""

        return bool(self._audio_buffer)

    def mark_endpoint(self) -> None:
        """将非空音频段标记为结束。"""

        if self._audio_buffer:
            self._endpoint = True

    def reset(self) -> None:
        """清空当前识别段状态。"""

        self._audio_buffer = []
        self._pending_samples = 0
        self._total_samples = 0
        self._last_voice_sample = 0
        self._has_voice = False
        self._text = ""
        self._endpoint = False
        self._finalized = False

    def new_stream(self) -> None:
        """开始新的识别段。"""

        self.reset()

    @property
    def model_loaded(self) -> bool:
        """返回 FunASR 模型是否已初始化。"""

        return self._recognizer is not None

    @staticmethod
    def _looks_like_local_path(value: str) -> bool:
        """判断模型字段是否是本地文件系统路径。"""

        return bool(
            value.startswith((".", "/", "\\", "~"))
            or ":\\" in value
            or Path(value).is_absolute()
        )

    @staticmethod
    def _normalize_device(device: str) -> str:
        """解析 FunASR 的 auto、GPU 和 CUDA 设备别名。"""

        value = device.strip().lower()
        if value in {"", "auto"}:
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if value in {"gpu", "cuda"}:
            return "cuda:0"
        return value

    def _load_model(self) -> Any:
        """首次推理时导入 FunASR 并创建模型。"""

        if self._recognizer is not None:
            return self._recognizer
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("缺少依赖 funasr，无法加载语音识别模型") from exc
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "缺少依赖 transformers；Fun-ASR-Nano 需要 transformers 和 tokenizers"
            ) from exc
        asr = self._config.asr
        remote_code = (
            self._resolve_remote_code(asr.model, asr.remote_code)
            if asr.trust_remote_code
            else asr.remote_code
        )
        if asr.trust_remote_code and remote_code:
            self._import_remote_code(remote_code)
        kwargs: dict[str, Any] = {
            "model": asr.model,
            "device": self._normalize_device(asr.device),
            "hub": asr.hub,
            "disable_update": True,
            "check_latest": False,
            "trust_remote_code": asr.trust_remote_code,
            "remote_code": remote_code,
            "disable_pbar": True,
            "log_level": "WARNING",
        }
        if asr.vad_model.strip():
            kwargs["vad_model"] = asr.vad_model.strip()
            kwargs["vad_kwargs"] = {
                "max_single_segment_time": asr.max_single_segment_time
            }
        try:
            self._recognizer = AutoModel(**kwargs)
        except AssertionError as exc:
            if "FunASRNano is not registered" not in str(exc):
                raise
            raise RuntimeError(
                "当前 FunASR 未注册 Fun-ASR-Nano；请确认安装版本及模型仓库代码配置"
            ) from exc
        return self._recognizer

    @staticmethod
    def _resolve_remote_code(model: str, remote_code: str) -> str:
        """解析本地模型代码路径或保留远程 URL。"""

        value = remote_code.strip()
        if not value or value.startswith(("http://", "https://")):
            return value
        path = Path(value)
        if path.is_file():
            return path.resolve().as_posix()
        repo_candidate = Path(__file__).resolve().parents[3] / value
        if repo_candidate.is_file():
            return repo_candidate.resolve().as_posix()
        model_path = Path(model)
        if model_path.exists():
            candidate = model_path / value
            if candidate.is_file():
                return candidate.resolve().as_posix()
        if value in {"model", "model.py", "./model.py", ".\\model.py"}:
            raise RuntimeError(
                "FunASR 配置指定的 remote_code 文件不存在: "
                f"{remote_code}（未在 anima_voice 中找到该文件）"
            )
        raise RuntimeError(f"FunASR remote_code 文件不存在: {remote_code}")

    @staticmethod
    def _import_remote_code(remote_code: str) -> None:
        """导入 FunASR 模型代码，兼容无动态导入工具的版本。"""

        try:
            from funasr.utils.dynamic_import import import_module_from_path
        except ImportError as exc:
            if remote_code.startswith(("http://", "https://")):
                raise RuntimeError("当前 FunASR 版本不支持远程 remote_code 导入") from exc
            path = Path(remote_code)
            if not path.is_file():
                raise RuntimeError(f"FunASR remote_code 文件不存在: {remote_code}")
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.append(parent)
            importlib.import_module(path.stem)
        else:
            import_module_from_path(remote_code)

    def _trim_audio_buffer(self) -> None:
        """限制单段音频的最大缓存时长。"""

        import numpy as np

        max_samples = max(
            1,
            int(
                self._config.audio.sample_rate
                * self._config.asr.max_audio_window_ms
                / 1000
            ),
        )
        total = sum(int(chunk.size) for chunk in self._audio_buffer)
        while self._audio_buffer and total > max_samples:
            first = self._audio_buffer[0]
            overflow = total - max_samples
            if overflow >= first.size:
                self._audio_buffer.pop(0)
                total -= int(first.size)
                continue
            self._audio_buffer[0] = np.ascontiguousarray(first[int(overflow) :])
            break

    def _run_inference(self, *, is_final: bool) -> None:
        """调用 FunASR 推理并更新最后一次非空文本。"""

        if not self._audio_buffer:
            return
        import numpy as np
        import torch

        samples = np.ascontiguousarray(np.concatenate(self._audio_buffer), dtype=np.float32)
        if samples.size == 0:
            return
        result = self._load_model().generate(
            **self._build_generate_kwargs(samples, torch, is_final=is_final)
        )
        text = self._extract_text(result)
        if text:
            self._text = text

    def _build_generate_kwargs(
        self, samples: Any, torch_module: Any, *, is_final: bool
    ) -> dict[str, Any]:
        """构造与所选模型类型匹配的 generate 参数。"""

        asr = self._config.asr
        kwargs: dict[str, Any] = {
            "input": torch_module.from_numpy(samples),
            "cache": {},
            "batch_size_s": asr.batch_size_s,
            "disable_pbar": True,
        }
        if asr.model_type == "funasr_nano":
            kwargs.update(
                {
                    "max_length": asr.max_length,
                    "llm_dtype": asr.llm_dtype,
                    "ctc_only": asr.ctc_only,
                    "is_final": is_final,
                }
            )
        return kwargs

    def _extract_text(self, result: Any) -> str:
        """从 FunASR 的多种输出结构中提取识别文本。"""

        if isinstance(result, str):
            return self._postprocess_text(result)
        if isinstance(result, tuple):
            return self._extract_text(result[0] if result else "")
        if isinstance(result, dict):
            return self._postprocess_text(str(result.get("text") or ""))
        if isinstance(result, list):
            parts: list[str] = []
            for item in result:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    ctc_text = str(item.get("ctc_text") or "").strip()
                    if (ctc_text and is_obviously_invalid_asr_text(text).accepted) or not text:
                        text = ctc_text
                    if text:
                        parts.append(self._postprocess_text(text))
                elif isinstance(item, str) and item.strip():
                    parts.append(self._postprocess_text(item))
            return "".join(parts).strip()
        return ""

    def _postprocess_text(self, text: str) -> str:
        """移除 SenseVoice 输出中的任务标记。"""

        normalized = text.strip()
        if not normalized or self._config.asr.model_type != "sensevoice_small":
            return normalized
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError:
            return normalized
        return str(rich_transcription_postprocess(normalized)).strip()


__all__ = ["StreamingRecognizer"]