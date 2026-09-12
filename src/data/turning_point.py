"""
变盘检测模块
基于 ZigZag + 量能确认检测 TOP / BOTTOM 变盘节点。

仅依赖 numpy + pandas，向量化计算，不逐行循环。
"""

import numpy as np
import pandas as pd


def limit_pct_for_symbol(symbol, is_st=False):
    """
    根据板块返回涨跌停阈值（近似）。
    主板 0.09；创业板(300/301)、科创板(688)、ST 均为 0.19。
    ST 需显式传 is_st=True（仅从 symbol 无法识别 ST）。
    """
    sym = str(symbol).strip().zfill(6)
    if is_st or sym.startswith("300") or sym.startswith("301") or sym.startswith("688"):
        return 0.19
    return 0.09


def detect_turning_points(df, up_pct=0.09, dn_pct=-0.09, vol_mult=2.0):
    """
    输入：单只股票OHLCV DataFrame，DatetimeIndex，列含Open/High/Low/Close/Volume
    输出：列表 [{"date": Timestamp, "type": "TOP"|"BOTTOM", "confidence": float}, ...]

    判定（i 为变盘节点，前后各留 1 天，不检测首尾）：
      TOP   : 前一日涨幅 > up_pct  且 后一日跌幅 < dn_pct 且 当日成交量 > 20日均量 * vol_mult
      BOTTOM: 前一日跌幅 < dn_pct  且 后一日涨幅 > up_pct 且 当日成交量 > 20日均量 * vol_mult
    confidence = min(当日量比 / 3.0, 1.0)
    """
    if df is None or len(df) < 3:
        return []
    if "Close" not in df.columns:
        return []

    close_raw = df["Close"].astype(float).to_numpy()
    vol_raw = df["Volume"].astype(float).to_numpy() if "Volume" in df.columns else np.ones(len(df))
    n = len(close_raw)

    # 停牌 / NaN 处理：收盘价前向填充（消除跨停牌的虚假涨跌），成交量 NaN 置 0
    close_ff = pd.Series(close_raw).ffill().bfill().fillna(0.0).to_numpy()
    vol = np.nan_to_num(vol_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 日收益率（向量化）
    ret = np.zeros(n, dtype=float)
    np.divide(close_ff[1:] - close_ff[:-1], close_ff[:-1],
              out=ret[1:], where=close_ff[:-1] != 0)

    # 20 日均量（向量化）
    vol_ma = pd.Series(vol).rolling(20, min_periods=1).mean().to_numpy()
    vol_ratio = np.divide(vol, vol_ma + 1e-8, out=np.zeros(n, dtype=float), where=vol_ma > 0)

    # 节点 i 取值范围 [1, n-2]
    prev_ret = ret[0:-2]        # ret[i-1]
    next_ret = ret[2:]          # ret[i+1]
    vol_i = vol[1:-1]           # volume[i]
    vol_ma_i = vol_ma[1:-1]
    vol_ratio_i = vol_ratio[1:-1]
    dates_i = df.index[1:-1]

    # 三个交易日收盘价均需有效（避免跨停牌误判）
    valid_close = np.isfinite(close_raw)
    valid_trio = valid_close[0:-2] & valid_close[1:-1] & valid_close[2:]

    top_mask = (prev_ret > up_pct) & (next_ret < dn_pct) & (vol_i > vol_ma_i * vol_mult) & valid_trio
    bot_mask = (prev_ret < dn_pct) & (next_ret > up_pct) & (vol_i > vol_ma_i * vol_mult) & valid_trio

    conf = np.clip(vol_ratio_i / 3.0, 0.0, 1.0)

    results = []
    for pos in np.where(top_mask)[0]:
        results.append({"date": dates_i[pos], "type": "TOP", "confidence": float(conf[pos])})
    for pos in np.where(bot_mask)[0]:
        results.append({"date": dates_i[pos], "type": "BOTTOM", "confidence": float(conf[pos])})

    results.sort(key=lambda r: r["date"])
    return results


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    close = np.array([10.0, 10.0, 10.0, 11.5, 11.5, 10.1, 10.1, 8.6, 8.6, 9.6, 9.6, 9.6])
    volume = np.array([10, 10, 10, 10, 200, 10, 10, 10, 200, 10, 10, 10], dtype=float)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.001,
        "Low": close * 0.999,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    pivots = detect_turning_points(df)
    print("detected pivots:")
    for p in pivots:
        print("  ", p["date"].date(), p["type"], round(p["confidence"], 3))

    print("limit_pct 000001:", limit_pct_for_symbol("000001"))
    print("limit_pct 300001:", limit_pct_for_symbol("300001"))
    print("limit_pct 688001:", limit_pct_for_symbol("688001"))
    print("limit_pct 600001 is_st:", limit_pct_for_symbol("600001", is_st=True))