"""
阶段片段特征向量提取（简单版 10 维）。

维度：
  0  5日动量
  1  10日动量
  2  20日动量
  3  量比
  4  换手分位
  5  60日区间位置
  6  涨停次数
  7  跌停次数
  8  波动率
  9  趋势方向（+1 / -1）
"""

import numpy as np
import pandas as pd

LIMIT_PCT = 0.095  # 涨/跌停近似阈值


def extract_stage_features(segment_df):
    """输入单片段 OHLCV DataFrame，输出长度 10 的 float32 向量。"""
    if segment_df is None or len(segment_df) == 0:
        return np.zeros(10, dtype=np.float32)

    close = segment_df["Close"].astype(float)
    high = segment_df["High"].astype(float) if "High" in segment_df.columns else close
    low = segment_df["Low"].astype(float) if "Low" in segment_df.columns else close
    volume = segment_df["Volume"].astype(float) if "Volume" in segment_df.columns else pd.Series(1.0, index=segment_df.index)

    n = len(close)

    # 1-3 动量
    mom5 = float(close.iloc[-1] / close.iloc[max(0, n - 6)] - 1.0)
    mom10 = float(close.iloc[-1] / close.iloc[max(0, n - 11)] - 1.0)
    mom20 = float(close.iloc[-1] / close.iloc[max(0, n - 21)] - 1.0)

    # 4 量比：最新量 / 20 日均量
    vol_ma = volume.rolling(20, min_periods=5).mean()
    vol_ratio = float(volume.iloc[-1] / (vol_ma.iloc[-1] + 1e-8))

    # 5 换手分位（Turnover 列缺失时置 0.5）
    turnover_pct = 0.5
    if "Turnover" in segment_df.columns:
        t = segment_df["Turnover"].astype(float).dropna()
        if len(t) > 0:
            last = float(t.iloc[-1])
            turnover_pct = float((t <= last).mean())

    # 6 60日（或全段）区间位置
    win = min(60, n)
    hh = float(high.iloc[-win:].max())
    ll = float(low.iloc[-win:].min())
    pos60 = float((close.iloc[-1] - ll) / (hh - ll + 1e-8))

    # 7-8 涨/跌停次数
    ret = close.pct_change().fillna(0.0)
    limit_up = int((ret >= LIMIT_PCT).sum())
    limit_down = int((ret <= -LIMIT_PCT).sum())

    # 9 波动率：日收益率标准差
    volatility = float(ret.std())

    # 10 趋势方向
    trend = 1.0 if float(close.iloc[-1]) > float(close.iloc[0]) else -1.0

    vec = np.array([
        np.tanh(mom5 * 5.0),
        np.tanh(mom10 * 3.0),
        np.tanh(mom20 * 2.0),
        np.tanh(vol_ratio * 0.5),
        turnover_pct,
        pos60,
        min(limit_up / 3.0, 1.0),
        min(limit_down / 3.0, 1.0),
        np.tanh(volatility * 10.0),
        trend,
    ], dtype=np.float32)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    close = np.linspace(10, 15, 30)
    volume = np.linspace(100, 300, 30)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    fv = extract_stage_features(df)
    print("feature vector:", fv)
    print("shape:", fv.shape, "dtype:", fv.dtype)