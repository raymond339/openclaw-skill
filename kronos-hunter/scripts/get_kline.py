#!/usr/bin/env python3
"""
获取单只股票的日K线数据并计算技术指标
用法: python3 get_kline.py <股票代码> [天数]
示例: python3 get_kline.py sh.600588 30
"""

import sys
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta


def get_kline(stock_code, days=30):
    lg = bs.login()
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    
    data_list = []
    while rs.error_code == "0" and rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout()
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 计算指标
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    
    # 动能
    df["mom5"] = df["close"].pct_change(5) * 100
    df["mom3"] = df["close"].pct_change(3) * 100
    
    # 量比
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"]
    
    # 涨停价
    df["prev_close"] = df["close"].shift(1)
    df["limit_up"] = (df["prev_close"] * 1.1).round(2)
    df["dist_to_limit"] = (df["limit_up"] - df["close"]) / df["limit_up"] * 100
    
    return df.dropna()


def main():
    if len(sys.argv) < 2:
        print("用法: python3 get_kline.py <股票代码> [天数]")
        print("示例: python3 get_kline.py sh.600588 30")
        sys.exit(1)
    
    stock_code = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    df = get_kline(stock_code, days)
    
    print(f"\n📊 {stock_code} K线数据（最近{len(df)}个交易日）")
    print("=" * 80)
    print(df[["date", "close", "MA5", "MA10", "mom5", "vol_ratio", "dist_to_limit"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
