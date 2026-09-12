"""
Episode 切分与阶段打标模块。

- split_episodes：根据变盘节点把整段行情切成多个片段（片段起止点为整数索引）。
- label_stage：对单个片段打阶段标签（PRE_BOOST / BOOSTING / TOP / CRASH）。
"""

import numpy as np
import pandas as pd

LIMIT_PCT = 0.095  # 涨/跌停近似阈值（覆盖主板10%板；创业板/科创板20%板的大涨大跌日同样 >= 0.095）


def _max_consecutive(mask):
    """向量化计算布尔序列中最长连续 True 的长度。"""
    a = np.asarray(mask, dtype=bool)
    if a.size == 0:
        return 0
    padded = np.concatenate(([0], a.view(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def _find_index(df, date):
    """把变盘节点日期映射为 DataFrame 中的整数行号，找不到返回 None。"""
    try:
        ts = pd.Timestamp(date)
        pos = df.index.get_loc(ts)
        if isinstance(pos, slice):
            pos = pos.start
        if isinstance(pos, (list, np.ndarray)):
            return int(pos[0]) if len(pos) else None
        return int(pos)
    except Exception:
        return None


def split_episodes(pivots, df, min_span=3):
    """
    输入：变盘节点列表 [{date, type, confidence}, ...] + OHLCV DataFrame + 最小间隔。
    输出：片段列表 [{"start_idx": int, "end_idx": int, "start_date": ..., "end_date": ...}, ...]

    规则：
      - 节点归后段起点（该节点所在日期成为下一片段的第一天）。
      - 相邻同向节点间隔 < min_span 时，保留 confidence 高者。
      - 无节点时返回单个整段片段。
    start_idx / end_idx 均为闭区间（切片用 df.iloc[start_idx:end_idx+1]）。
    """
    n = len(df)
    if n == 0:
        return []

    node_items = []
    for p in (pivots or []):
        pos = _find_index(df, p.get("date"))
        if pos is None:
            continue
        node_items.append({
            "idx": pos,
            "type": p.get("type"),
            "confidence": float(p.get("confidence", 0.0) or 0.0),
        })

    if not node_items:
        return [{
            "start_idx": 0,
            "end_idx": n - 1,
            "start_date": df.index[0],
            "end_date": df.index[-1],
        }]

    node_items.sort(key=lambda x: x["idx"])

    # 相邻同向节点合并：间隔 < min_span 保留 confidence 高者
    merged = []
    for item in node_items:
        if merged and item["type"] == merged[-1]["type"] and (item["idx"] - merged[-1]["idx"]) < min_span:
            if item["confidence"] > merged[-1]["confidence"]:
                merged[-1] = item
        else:
            merged.append(item)

    # 切分点：0 + 各节点索引；节点必须落在 (0, n-1) 内，否则忽略
    cut_points = sorted({m["idx"] for m in merged if 0 < m["idx"] < n})
    boundaries = [0] + cut_points

    segments = []
    for k in range(len(boundaries)):
        start = boundaries[k]
        end = (boundaries[k + 1] - 1) if (k + 1) < len(boundaries) else (n - 1)
        if end < start:
            continue
        segments.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "start_date": df.index[start],
            "end_date": df.index[end],
        })
    return segments


def label_stage(segment_df):
    """
    输入：单个片段的 OHLCV DataFrame。
    输出：stage 字符串，取值 PRE_BOOST / BOOSTING / TOP / CRASH。

    判定优先级：CRASH > TOP > BOOSTING > PRE_BOOST（PRE_BOOST 为兜底标签）。

      CRASH  : 顶后跌幅 >= 15%（峰值回撤 <= -15%），或连续跌停 >= 2。
      TOP    : 触顶后回撤 > 5%（回撤 <= -5%），且量比最大 > 3，且平均上影线占比 > 0.30。
      BOOSTING: 区间涨幅 >= 30%，且出现过涨停，且平均量比 > 2。
      PRE_BOOST: 兜底（区间涨幅 < 20%、量比 < 1.5、价格在区间下 30% 视为典型蓄势）。
    """
    if segment_df is None or len(segment_df) == 0:
        return "PRE_BOOST"

    close = segment_df["Close"].astype(float).to_numpy()
    high = segment_df["High"].astype(float).to_numpy() if "High" in segment_df.columns else close.copy()
    low = segment_df["Low"].astype(float).to_numpy() if "Low" in segment_df.columns else close.copy()
    open_ = segment_df["Open"].astype(float).to_numpy() if "Open" in segment_df.columns else close.copy()
    volume = segment_df["Volume"].astype(float).to_numpy() if "Volume" in segment_df.columns else np.ones(len(close))

    n = len(close)
    seg_ret = float(close[-1] / close[0] - 1.0) if close[0] != 0 else 0.0

    # 平均/最大量比（对自身 20 日均量）
    vol_ma = pd.Series(volume).rolling(20, min_periods=3).mean().to_numpy()
    daily_vol_ratio = volume / (vol_ma + 1e-8)
    vol_ratio_mean = float(np.nanmean(daily_vol_ratio))
    vol_ratio_max = float(np.nanmax(daily_vol_ratio))

    # 日收益率与涨跌停
    ret = np.zeros(n, dtype=float)
    if n > 1:
        np.divide(close[1:] - close[:-1], close[:-1], out=ret[1:], where=close[:-1] != 0)
    limit_up = int(np.sum(ret >= LIMIT_PCT))
    max_consec_limit_down = _max_consecutive(ret <= -LIMIT_PCT)

    # 峰值回撤
    drawdown = float(close[-1] / (np.max(np.maximum.accumulate(close)) + 1e-8) - 1.0)

    # 上影线占比（平均）
    upper_shadow = (high - np.maximum(open_, close)) / (high - low + 1e-8)
    upper_shadow_mean = float(np.nanmean(upper_shadow))

    if drawdown <= -0.15 or max_consec_limit_down >= 2:
        return "CRASH"
    if drawdown <= -0.05 and vol_ratio_max >= 3.0 and upper_shadow_mean > 0.30:
        return "TOP"
    if seg_ret >= 0.30 and limit_up >= 1 and vol_ratio_mean > 2.0:
        return "BOOSTING"
    return "PRE_BOOST"


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    # 构造一段“蓄势 -> 拉升 -> 见顶 -> 暴跌”的合成行情
    close = np.concatenate([
        np.linspace(10, 10.5, 10),        # 蓄势
        np.linspace(10.5, 18, 10),        # 拉升（大幅上涨）
        np.array([18.0, 18.2, 18.0, 17.2, 16.4]),  # 见顶回落
        np.linspace(16.4, 11.0, 15),      # 暴跌
    ])
    close = close[:len(dates)]
    volume = np.where(np.arange(len(close)) >= 30, 100, 20).astype(float)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    # 用简单节点示例演示切分（此处手动给两个节点）
    fake_pivots = [
        {"date": dates[10], "type": "BOTTOM", "confidence": 0.8},
        {"date": dates[25], "type": "TOP", "confidence": 0.9},
    ]
    segments = split_episodes(fake_pivots, df)
    print("segments:")
    for s in segments:
        sub = df.iloc[s["start_idx"]:s["end_idx"] + 1]
        print(f'  {s["start_date"].date()} ~ {s["end_date"].date()} '
              f'({s["start_idx"]}-{s["end_idx"]}, len={len(sub)}) -> {label_stage(sub)}')

    # 无节点情况
    no_node = split_episodes([], df)
    print("no-node segment label:", label_stage(df.iloc[no_node[0]["start_idx"]:no_node[0]["end_idx"] + 1]))