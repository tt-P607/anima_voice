"""anima_voice 测试的项目路径与运行状态隔离。"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(autouse=True)
async def reset_call_state() -> AsyncIterator[None]:
    """隔离模块级通话状态。"""

    from plugins.anima_voice.runtime import call_state

    await call_state.clear_active_call()
    yield
    await call_state.clear_active_call()
