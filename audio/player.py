"""以独立 OutputStream 串行播放 WAV 和增量 PCM。"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd
import soundfile as sf

from .._internal_compat import create_task


class AudioPlayer:
    """使用本插件独立锁和设备流输出 WAV 与 PCM 音频。"""

    def __init__(
        self, output_device: str = "", loudness_target_dbfs: float = -14.0
    ) -> None:
        """解析目标设备并初始化串行锁。"""

        self.output_device = output_device.strip()
        self.loudness_target_dbfs = loudness_target_dbfs
        self._lock = asyncio.Lock()
        self._stop_event = threading.Event()
        self._closed = False
        self._pcm_task: asyncio.Task[object] | None = None

    async def play_audio(self, audio_data: bytes) -> None:
        """在独立线程中解码并完整播放一个 WAV。"""

        if not audio_data:
            return
        async with self._lock:
            if self._closed:
                raise RuntimeError("音频播放器已关闭")
            self._stop_event.clear()
            handle = create_task(
                asyncio.to_thread(self._play_sync, audio_data),
                f"anima_voice.audio.{id(self)}",
            )
            try:
                await asyncio.shield(handle.task)
            except asyncio.CancelledError:
                self._stop_event.set()
                try:
                    await asyncio.shield(handle.task)
                finally:
                    raise

    async def close(self) -> None:
        """停止本插件输出并等待音频线程释放设备。"""

        self._closed = True
        await self.stop()

    async def stop(self) -> None:
        """停止当前自有播放，播放器仍可用于后续告别语。"""

        self._stop_event.set()
        task = self._pcm_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with self._lock:
            return

    async def play_pcm_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        sample_rate: int,
        channels: int = 1,
        sample_format: str = "s16le",
    ) -> int:
        """在单个设备流中增量播放 PCM，返回写入帧数，取消时等待线程退出。"""

        if sample_rate <= 0 or channels <= 0 or sample_format != "s16le":
            raise ValueError("PCM 必须是正采样率、正声道数的 s16le")
        async with self._lock:
            if self._closed:
                raise RuntimeError("音频播放器已关闭")
            self._stop_event.clear()
            output: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
            errors: list[BaseException] = []
            frames_written = 0
            frame_bytes = 2 * channels
            loop = asyncio.get_running_loop()
            finished: asyncio.Future[None] = loop.create_future()
            self._pcm_task = asyncio.current_task()

            def mark_finished() -> None:
                """将线程结束通知提交给事件循环。"""

                if not finished.done():
                    finished.set_result(None)

            def write_pcm() -> None:
                """独占设备流，原序写入已对齐的 PCM 块。"""

                nonlocal frames_written
                try:
                    with sd.OutputStream(
                        samplerate=sample_rate, channels=channels,
                        dtype="int16", device=self._resolve_device(),
                    ) as stream:
                        while not self._stop_event.is_set():
                            try:
                                block = output.get(timeout=0.05)
                            except queue.Empty:
                                continue
                            if block is None:
                                return
                            samples = np.frombuffer(block, dtype="<i2").reshape(-1, channels)
                            stream.write(samples)
                            frames_written += len(samples)
                        stream.abort()
                except (ValueError, RuntimeError, OSError, sd.PortAudioError) as error:
                    errors.append(error)
                finally:
                    loop.call_soon_threadsafe(mark_finished)

            writer = threading.Thread(target=write_pcm, name="anima-voice-pcm", daemon=True)
            writer.start()

            async def enqueue(block: bytes | None) -> None:
                """等待有界设备队列空间，同时检查取消和输出错误。"""

                while True:
                    if errors:
                        raise RuntimeError("PCM 输出失败") from errors[0]
                    if self._stop_event.is_set():
                        raise RuntimeError("PCM 输出已停止")
                    try:
                        output.put_nowait(block)
                        return
                    except queue.Full:
                        await asyncio.to_thread(self._stop_event.wait, 0.01)

            async def feed_pcm() -> None:
                """对齐采样帧，首块即交付，结束时拒绝不完整采样。"""

                pending = bytearray()
                received = False
                block_bytes = max(frame_bytes, sample_rate // 10 * frame_bytes)
                async for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("PCM 块必须是 bytes")
                    pending.extend(chunk)
                    usable = len(pending) - len(pending) % frame_bytes
                    while usable:
                        size = min(usable, block_bytes)
                        await enqueue(bytes(pending[:size]))
                        del pending[:size]
                        usable -= size
                        received = True
                if pending:
                    raise ValueError("PCM 流结束时存在不完整采样帧")
                if not received:
                    raise ValueError("PCM 流为空")
                await enqueue(None)

            feeder = create_task(feed_pcm(), f"anima_voice.pcm_feed.{id(self)}").task
            try:
                await asyncio.wait({feeder, finished}, return_when=asyncio.FIRST_COMPLETED)
                if errors:
                    raise RuntimeError("PCM 输出失败") from errors[0]
                await feeder
                await asyncio.to_thread(writer.join)
                if errors:
                    raise RuntimeError("PCM 输出失败") from errors[0]
                return frames_written
            finally:
                self._stop_event.set()
                if not feeder.done():
                    feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)
                try:
                    close = getattr(chunks, "aclose", None)
                    if callable(close):
                        await close()
                finally:
                    await asyncio.to_thread(writer.join)
                    self._pcm_task = None

    def _play_sync(self, audio_data: bytes) -> None:
        """解码、可选响度归一化并通过专属设备流播放。"""

        samples, sample_rate = sf.read(
            io.BytesIO(audio_data), dtype="float32", always_2d=True
        )
        if self._stop_event.is_set():
            return
        if self.loudness_target_dbfs != 0:
            rms = float(np.sqrt(np.mean(np.square(samples))))
            if rms > 0:
                target = 10 ** (self.loudness_target_dbfs / 20)
                samples = np.clip(samples * target / rms, -1.0, 1.0)
        device = self._resolve_device()
        with sd.OutputStream(
            samplerate=sample_rate, channels=samples.shape[1], device=device
        ) as stream:
            block_size = max(1, sample_rate // 10)
            for offset in range(0, len(samples), block_size):
                if self._stop_event.is_set():
                    stream.abort()
                    return
                stream.write(samples[offset : offset + block_size])

    def _resolve_device(self) -> int | None:
        """按设备名和可选 host API 名解析输出设备。"""

        if not self.output_device:
            return None
        name, separator, api = self.output_device.partition("@")
        host_apis = sd.query_hostapis()
        for index, device in enumerate(sd.query_devices()):
            if not device["max_output_channels"] or name not in device["name"]:
                continue
            if (
                separator
                and api.lower() not in host_apis[device["hostapi"]]["name"].lower()
            ):
                continue
            return index
        raise ValueError(f"未找到音频输出设备：{self.output_device}")
