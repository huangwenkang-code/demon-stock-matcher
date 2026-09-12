"""
批量构建妖股生命周期数据湖。

流程：
  读取 data/raw/ 下所有 CSV
    -> detect_turning_points -> split_episodes -> label_stage -> extract_stage_features
    -> 写入 LifecycleStore（DuckDB）

用法：python scripts/build_lifecycle_lake.py
"""

import glob
import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.turning_point import detect_turning_points, limit_pct_for_symbol  # noqa: E402
from src.data.episode_splitter import split_episodes, label_stage  # noqa: E402
from src.data.lifecycle_store import LifecycleStore  # noqa: E402
from src.features.stage_features import extract_stage_features  # noqa: E402

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "lifecycle.duckdb")


def _normalize_columns(df):
    """把中文/大小写列名统一映射为 Open/High/Low/Close/Volume/Amount/Turnover。"""
    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("open", "开盘"):
            col_map[c] = "Open"
        elif lc in ("high", "最高"):
            col_map[c] = "High"
        elif lc in ("low", "最低"):
            col_map[c] = "Low"
        elif lc in ("close", "收盘", "收盘价"):
            col_map[c] = "Close"
        elif lc in ("volume", "成交量"):
            col_map[c] = "Volume"
        elif lc in ("amount", "成交额", "成交金额", "成交额(元)"):
            col_map[c] = "Amount"
        elif lc in ("turnover", "换手", "换手率"):
            col_map[c] = "Turnover"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def process_symbol(symbol, store):
    """处理单只股票，返回 (片段数, 变盘节点数)。"""
    path = os.path.join(DATA_RAW_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return 0, 0

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = _normalize_columns(df)
    if "Close" not in df.columns or "Volume" not in df.columns:
        return 0, 0

    df = df[~df.index.duplicated(keep="last")].sort_index()
    # 停牌等造成的空值行，仅剔除 Close/Volume 均为空的记录
    df = df[df["Close"].notna() & df["Volume"].notna()]
    if len(df) < 3:
        return 0, 0

    # 幂等重建：先清掉该 symbol 的旧记录
    store.delete_episode(symbol)

    limit = limit_pct_for_symbol(symbol)
    pivots = detect_turning_points(df, up_pct=limit, dn_pct=-limit)
    segments = split_episodes(pivots, df)

    seg_rows = []
    for seg in segments:
        sub = df.iloc[seg["start_idx"]:seg["end_idx"] + 1]
        seg_rows.append({
            "symbol": symbol,
            "stage": label_stage(sub),
            "start_date": seg["start_date"],
            "end_date": seg["end_date"],
            "start_idx": seg["start_idx"],
            "end_idx": seg["end_idx"],
            "feature_vec": extract_stage_features(sub),
            "valid_from": seg["end_date"],
        })

    store.insert_episode(symbol, symbol, df.index[0], df.index[-1], len(seg_rows))
    store.insert_turning_points(symbol, pivots)
    store.insert_segments(symbol, seg_rows)

    return len(seg_rows), len(pivots)


def main():
    csv_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*.csv")))
    if not csv_files:
        print(f"[warn] 未在 {DATA_RAW_DIR} 下找到任何 CSV")
        return

    total_episodes = 0
    total_segments = 0
    total_pivots = 0

    with LifecycleStore(DB_PATH) as store:
        for i, f in enumerate(csv_files, 1):
            symbol = os.path.splitext(os.path.basename(f))[0].zfill(6)
            try:
                ns, npiv = process_symbol(symbol, store)
            except Exception as e:
                print(f"[skip] {symbol}: {e}")
                continue
            total_episodes += 1
            total_segments += ns
            total_pivots += npiv
            print(f"[{i}/{len(csv_files)}] {symbol}: pivots={npiv} segments={ns}")

    print(f"done: episodes={total_episodes} segments={total_segments} pivots={total_pivots}")


if __name__ == "__main__":
    main()