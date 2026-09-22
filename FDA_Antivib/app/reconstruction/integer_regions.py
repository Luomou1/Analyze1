"""EXE High 2G 的整数域连接：0x9EC930、0x9EE3C0、0x9EE620、0x9EEA50。

保留扫描种子顺序、直方图分桶、桶内后进先出、8 邻域和冲突拒绝。
不能以通用质量堆或最小二乘空间展开替换这些顺序相关分支。
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

INVALID: Final = 0x7FFFFFF8
BUCKET_COUNT: Final = 30
MIN_REGION: Final = 7


@dataclass(frozen=True, slots=True)
class ConnectionResult:
    connected: NDArray[np.float64]
    merit: NDArray[np.float64]
    labels: NDArray[np.int64]


def connection_merit(
    values: NDArray[np.int64], valid: NDArray[np.bool_], period: int,
) -> NDArray[np.float64]:
    """0x9EC930：有效邻点的周期距离均值，至少三个邻点，向零截断。"""
    height, width = values.shape
    total = np.zeros(values.shape, dtype=np.int64)
    count = np.zeros(values.shape, dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            dst = (slice(max(0, -dy), min(height, height - dy)),
                   slice(max(0, -dx), min(width, width - dx)))
            src = (slice(max(0, dy), min(height, height + dy)),
                   slice(max(0, dx), min(width, width + dx)))
            good = valid[dst] & valid[src]
            delta = np.abs(values[dst] - values[src])
            distance = np.abs(np.where(delta > period // 2, period - delta, delta))
            total[dst] += np.where(good, distance, 0)
            count[dst] += good
    result = np.full(values.shape, np.inf)
    np.divide(total, count, out=result, where=valid & (count >= 3))
    return np.trunc(result)


def connect_regions(
    values: NDArray[np.int64], valid: NDArray[np.bool_], *, period: int, noise_percent: float,
) -> ConnectionResult:
    """0xB41C10 选择的 conflict=1 分支；冲突像素不重新排队。"""
    if values.ndim != 2 or values.shape != valid.shape or period <= 0:
        raise ValueError("Connection requires equally shaped 2-D maps and a positive period.")
    if not np.isfinite(noise_percent) or not 0 < noise_percent <= 100:
        raise ValueError("Connection noise threshold must be in (0, 100].")
    merit = connection_merit(values, valid, period)
    cutoff = min(int(period * noise_percent / 100), 0x17FFF)
    eligible = valid & (merit < cutoff)
    quality = np.where(eligible, merit, 0).astype(np.int64)
    histogram = np.bincount(quality[eligible], minlength=cutoff)
    target = max(1, int(eligible.sum()) // BUCKET_COUNT)
    mapping = np.zeros(cutoff, dtype=np.int64)
    bucket, accumulated = 0, 0
    for value in range(cutoff):
        accumulated += int(histogram[value])
        mapping[value] = bucket
        if accumulated >= target:
            bucket += 1
            accumulated = 0
    buckets: list[list[tuple[int, int]]] = [[] for _ in range(bucket + 1)]
    queued = np.zeros(values.shape, dtype=bool)
    orders = np.full(values.shape, INVALID, dtype=np.int64)
    labels = np.full(values.shape, -1, dtype=np.int64)
    height, width = values.shape
    region = 0
    for sy in range(1, height - 1):
        for sx in range(1, width - 1):
            if not eligible[sy, sx] or queued[sy, sx]:
                continue
            orders[sy, sx] = 0
            labels[sy, sx] = region
            queued[sy, sx] = True
            members = [(sy, sx)]
            for yy in range(sy - 1, sy + 2):
                for xx in range(sx - 1, sx + 2):
                    if eligible[yy, xx] and not queued[yy, xx]:
                        buckets[int(mapping[quality[yy, xx]])].append((yy, xx))
                        queued[yy, xx] = True
            current = 0
            while current < len(buckets):
                if not buckets[current]:
                    current += 1
                    continue
                y, x = buckets[current].pop()
                candidates: list[int] = []
                for yy in range(max(0, y - 1), min(height, y + 2)):
                    for xx in range(max(0, x - 1), min(width, x + 2)):
                        if orders[yy, xx] == INVALID:
                            continue
                        delta = int(values[yy, xx] - values[y, x])
                        correction = 0 if abs(delta) < period // 2 else (1 if delta > 0 else -1)
                        candidates.append(int(orders[yy, xx]) + correction)
                if not candidates or any(order != candidates[0] for order in candidates):
                    continue
                orders[y, x] = candidates[0]
                labels[y, x] = region
                members.append((y, x))
                for yy in range(max(0, y - 1), min(height, y + 2)):
                    for xx in range(max(0, x - 1), min(width, x + 2)):
                        if eligible[yy, xx] and not queued[yy, xx]:
                            next_bucket = int(mapping[quality[yy, xx]])
                            buckets[next_bucket].append((yy, xx))
                            current = min(current, next_bucket)
                            queued[yy, xx] = True
            if len(members) < MIN_REGION:
                for y, x in members:
                    orders[y, x] = INVALID
                    labels[y, x] = -1
            else:
                region += 1
    connected = np.where(orders != INVALID, values + period * orders, np.nan)
    return ConnectionResult(connected, merit, labels)
