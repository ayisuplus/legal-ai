"""审计追踪：只追加事件流，支撑可追溯与越界率统计（启动书 10.2 / 十一章）。

导出为 JSONL，可直接作为能力越界 Benchmark（实验四/五）的数据源。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceEvent:
    seq: int
    event: str          # request_received / plan_created / tool_called /
                        # tool_returned / guard_decision / unit_emitted / terminated
    payload: dict
    ts: float = field(default_factory=time.time)


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(self, event: str, **payload) -> TraceEvent:
        e = TraceEvent(seq=len(self._events), event=event, payload=payload)
        self._events.append(e)
        return e

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def export_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(
                    {"seq": e.seq, "event": e.event, "ts": e.ts, "payload": e.payload},
                    ensure_ascii=False,
                ) + "\n")
        return path
