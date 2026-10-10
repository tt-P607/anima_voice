"""本机麦克风音频采集。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.app.plugin_system.api.log_api import get_logger

from .._internal_compat import create_task

logger = get_logger("anima_voice.asr.audio")


class MicrophoneAudioSource:
    """将 sounddevice 输入转换为有界单声道音频块队列。"""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        device: str,
        block_size: int,
        queue_max_chunks: int,
    ) -> None:
        """初始化音频源，不打开设备。"""

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device.strip() or None
        self.block_size = block_size
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_max_chunks)
        self._stream: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._dropped_chunks = 0
        self._last_status_warning_at = 0.0

    async def start(self) -> None:
        """惰性导入 sounddevice 并启动输入流。"""

        if self._running:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("缺少依赖 sounddevice，无法采集麦克风") from exc
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
            device=self._coerce_device(self.device),
            callback=self._on_audio,
        )
        self._stream.start()
        self._running = True
        actual_rate = getattr(self._stream, "samplerate", self.sample_rate)
        if int(actual_rate) != int(self.sample_rate):
            logger.warning(
                f"麦克风实际采样率与配置不一致: requested={self.sample_rate}, actual={actual_rate}"
            )

    async def stop(self) -> None:
        """停止并释放输入设备，清空待处理音频。"""

        stream = self._stream
        self._stream = None
        self._running = False
        if stream is not None:
            try:
                await self._run_device_call(stream.stop)
            finally:
                await self._run_device_call(stream.close)
        while not self._queue.empty():
            self._queue.get_nowait()

    async def read(self) -> Any:
        """等待下一个音频块。"""

        return await self._queue.get()

    def drain_pending(self) -> list[Any]:
        """非阻塞取出队列中现有音频块。"""

        chunks: list[Any] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return chunks

    def is_running(self) -> bool:
        """返回输入流是否启动。"""

        return self._running

    def queue_size(self) -> int:
        """返回已排队音频块数量。"""

        return self._queue.qsize()

    def queue_capacity(self) -> int:
        """返回队列容量。"""

        return self._queue.maxsize

    def _on_audio(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        """在设备回调线程中转换并安全投递音频块。"""

        loop = self._loop
        if status and loop is not None:
            now = time.monotonic()
            if now - self._last_status_warning_at >= 2.0:
                self._last_status_warning_at = now
                loop.call_soon_threadsafe(logger.warning, f"麦克风输入状态异常: {status}")
        if loop is None or not self._running:
            return
        loop.call_soon_threadsafe(self._put_latest, self._to_mono(indata).copy())

    def _put_latest(self, samples: Any) -> None:
        """队列满时移除旧块，优先保留实时音频。"""

        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_chunks += 1
                if self._dropped_chunks == 1 or self._dropped_chunks % 20 == 0:
                    logger.warning(f"ASR 队列已满，累计丢弃旧音频块: {self._dropped_chunks}")
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(samples)

    @staticmethod
    async def _run_device_call(function: Any) -> None:
        """等待阻塞设备操作完成，即使调用方任务收到取消。"""

        task_info = create_task(
            asyncio.to_thread(function), "anima_voice.asr.device_call"
        )
        try:
            await asyncio.shield(task_info.task)
        except asyncio.CancelledError:
            await asyncio.gather(task_info.task, return_exceptions=True)
            raise

    @staticmethod
    def _to_mono(samples: Any) -> Any:
        """将设备音频转换为一维 float32 数组。"""

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少依赖 numpy，无法转换麦克风音频") from exc
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 2:
            if array.shape[1] == 1:
                return array[:, 0]
            return array.mean(axis=1, dtype=np.float32)
        return array.reshape(-1)

    @staticmethod
    def _coerce_device(device: str | None) -> str | int | None:
        """把纯数字设备标识转换为整数索引。"""

        value = str(device or "").strip()
        if not value:
            return None
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return value