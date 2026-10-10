"""通话阻塞操作期间的 stream watchdog 心跳。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


@asynccontextmanager
async def keep_watchdog_alive(
    stream_id: str, interval: float = 5.0
) -> AsyncIterator[None]:
    """在长操作期间通过框架任务管理器周期性喂 watchdog。"""

    from .._internal_compat import cancel_task, create_task, feed_watchdog

    stop = asyncio.Event()

    async def feed() -> None:
        """按间隔刷新目标流心跳。"""

        while not stop.is_set():
            feed_watchdog(stream_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    handle = create_task(feed(), f"anima_voice.heartbeat.{stream_id[:8]}")
    try:
        yield
    finally:
        stop.set()
        cancel_task(handle)
