from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_DROZY_TIMESTAMP_FORMAT = "%Y-%m-%d_%H.%M.%S.%f"


@dataclass(frozen=True)
class PvtSample:
    reaction_ms: float
    timestamp_seconds: float | None = None


@dataclass(frozen=True)
class PvtSummary:
    mean_reaction_ms: float | None
    lapse_rate: float
    valid_count: int
    false_start_count: int
    normal_count: int = 0
    lapse_count: int = 0


def _parse_timestamp(value: str, path: Path, line_number: int) -> datetime:
    try:
        return datetime.strptime(value, _DROZY_TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"PVT時刻を解析できません: {path}:{line_number}: {value!r}"
        ) from exc


def read_pvt(path: str | Path) -> list[PvtSample]:
    """DROZYの試験開始時刻と刺激・反応時刻の組からPVT標本を復元する。"""
    source = Path(path)
    lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), start=1
        )
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"PVTファイルが空です: {source}")

    origin_line, origin_text = lines[0]
    if ";" in origin_text:
        raise ValueError(
            f"PVT試験開始時刻は1列である必要があります: {source}:{origin_line}"
        )
    origin = _parse_timestamp(origin_text, source, origin_line)

    samples: list[PvtSample] = []
    previous_stimulus: datetime | None = None
    for line_number, line in lines[1:]:
        columns = [value.strip() for value in line.split(";")]
        if len(columns) != 2 or not all(columns):
            raise ValueError(
                f"PVT反応行は刺激時刻と反応時刻の2列である必要があります: "
                f"{source}:{line_number}"
            )
        stimulus = _parse_timestamp(columns[0], source, line_number)
        response = _parse_timestamp(columns[1], source, line_number)
        elapsed_seconds = (stimulus - origin).total_seconds()
        reaction_ms = (response - stimulus).total_seconds() * 1000.0
        if elapsed_seconds < 0:
            raise ValueError(f"PVT刺激時刻が試験開始前です: {source}:{line_number}")
        if reaction_ms < 0:
            raise ValueError(f"PVT反応時刻が刺激時刻より前です: {source}:{line_number}")
        if previous_stimulus is not None and stimulus <= previous_stimulus:
            raise ValueError(f"PVT刺激時刻が単調増加ではありません: {source}:{line_number}")
        samples.append(PvtSample(reaction_ms=reaction_ms, timestamp_seconds=elapsed_seconds))
        previous_stimulus = stimulus

    if not samples:
        raise ValueError(f"PVT反応がありません: {source}")
    return samples


def summarize_pvt(
    samples: list[PvtSample],
    *,
    false_start_ms: float = 100.0,
    lapse_ms: float = 500.0,
) -> PvtSummary:
    if false_start_ms < 0 or lapse_ms <= false_start_ms:
        raise ValueError("PVTしきい値は 0 <= false_start_ms < lapse_ms が必要です")
    false_starts = [sample for sample in samples if sample.reaction_ms < false_start_ms]
    normal = [
        sample.reaction_ms
        for sample in samples
        if false_start_ms <= sample.reaction_ms < lapse_ms
    ]
    lapses = [sample for sample in samples if sample.reaction_ms >= lapse_ms]
    valid_count = len(normal) + len(lapses)
    if valid_count == 0:
        raise ValueError("有効なPVT反応時間がありません")
    return PvtSummary(
        mean_reaction_ms=sum(normal) / len(normal) if normal else None,
        lapse_rate=len(lapses) / valid_count,
        valid_count=valid_count,
        false_start_count=len(false_starts),
        normal_count=len(normal),
        lapse_count=len(lapses),
    )


def summarize_pvt_windows(
    samples: list[PvtSample],
    *,
    window_seconds: float = 20.0,
    false_start_ms: float = 100.0,
    lapse_ms: float = 500.0,
) -> list[tuple[float, PvtSummary]]:
    """時刻付きPVTを固定窓へ集計する。時刻の無い形式ではセッション集計だけを使う。"""
    timed = [sample for sample in samples if sample.timestamp_seconds is not None]
    if not timed:
        return []
    last = max(float(sample.timestamp_seconds) for sample in timed)
    output: list[tuple[float, PvtSummary]] = []
    start = 0.0
    while start <= last:
        in_window = [
            sample
            for sample in timed
            if start <= float(sample.timestamp_seconds) < start + window_seconds
        ]
        if in_window:
            try:
                summary = summarize_pvt(in_window, false_start_ms=false_start_ms, lapse_ms=lapse_ms)
            except ValueError:
                pass
            else:
                output.append((start + window_seconds / 2.0, summary))
        start += window_seconds
    return output


def impairment_from_baseline(current: PvtSummary, baseline: PvtSummary) -> float:
    """PVT1に対するRTとlapse率の悪化量。正なら覚醒度低下方向。"""
    if baseline.mean_reaction_ms is None:
        raise ValueError("PVT1に通常反応がなく、PVT基準を作成できません")
    rt_change = 0.0
    if current.mean_reaction_ms is not None:
        rt_change = (current.mean_reaction_ms - baseline.mean_reaction_ms) / max(
            baseline.mean_reaction_ms, 1.0
        )
    lapse_change = current.lapse_rate - baseline.lapse_rate
    return float(rt_change + lapse_change)
