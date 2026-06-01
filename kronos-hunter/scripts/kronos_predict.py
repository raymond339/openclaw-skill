#!/usr/bin/env python3
"""
KRONOS K线预测脚本
- 实时价格：新浪API（字段[7]卖一价）
- 5分钟K线：baostock
- K线预测：KRONOS模型（NeoQuasar/Kronos-small）

依赖: baostock, torch, transformers
安装: pip3 install baostock torch transformers -q

用法: python3 kronos_predict.py <股票代码>
示例: python3 kronos_predict.py sz.300696
"""

import sys
import os

# 添加本地model路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess


def get_realtime_price(stock_code):
    """从新浪API获取实时价格（字段[7]卖一价最可靠）"""
    try:
        code = stock_code.replace(".", "")
        url = f"https://hq.sinajs.cn/list={code}"
        
        result = subprocess.run(
            ["curl", "-s", "--max-time", "5", url, "-H", "Referer: https://finance.sina.com.cn"],
            capture_output=True, timeout=10
        )
        
        try:
            output = result.stdout.decode("gbk").strip()
        except UnicodeDecodeError:
            output = result.stdout.decode("utf-8", errors="ignore").strip()
        
        start = output.find('"')
        end = output.rfind('"')
        if start == -1 or end == -1:
            return None
        
        data = output[start+1:end].split(",")
        if len(data) < 10:
            return None
        
        # 新浪字段: 0=名称, 1=当前价(有时滞后), 2=昨收, 3=今开, 4=最高, 5=最低, 7=卖一(实时参考价)
        current_price = float(data[7]) if data[7] and data[7] != '0' else float(data[1])
        
        return {
            "name": data[0],
            "current": current_price,
            "prev_close": float(data[2]) if data[2] else 0,
            "open": float(data[3]) if data[3] else 0,
            "high": float(data[4]) if data[4] else 0,
            "low": float(data[5]) if data[5] else 0,
            "volume": int(data[8]) if data[8] else 0,
        }
    except Exception as e:
        print(f"  实时价格获取失败: {e}")
        return None


def get_5min_kline(stock_code, days=3):
    """获取最近3天的5分钟K线数据（来自baostock）"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f"baostock登录失败: {lg.error_msg}")
        return None
    
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,time,code,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="5",
        adjustflag="3",
    )
    
    if rs is None:
        print("查询失败: rs is None")
        bs.logout()
        return None
    
    data_list = []
    try:
        while rs.next():
            data_list.append(rs.get_row_data())
    except Exception as e:
        print(f"读取数据异常: {e}")
        bs.logout()
        return None
    
    bs.logout()
    
    if not data_list:
        print(f"未获取到数据，start={start_date}, end={end_date}")
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 转换时间戳 - time格式: 093500000 -> 09:35:00
    time_str = df["time"].str[:2] + ":" + df["time"].str[2:4] + ":" + df["time"].str[4:6]
    df["datetime"] = pd.to_datetime(df["date"] + " " + time_str, format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)
    
    return df.tail(512)


def load_kronos_model():
    """加载KRONOS模型"""
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        raise ImportError("本地model模块未找到，请确保已将 Kronos 仓库的 model 目录复制到 skills/kronos-hunter/")
    
    print("📥 正在从 Hugging Face 加载 Kronos-small 模型...")
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用设备: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  使用设备: CPU")
    
    model = model.to(device)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    return predictor


def run_prediction(predictor, df, pred_len=24):
    """执行KRONOS预测"""
    lookback = min(len(df), 400)
    
    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume']].copy()
    x_df['amount'] = 0  # 5分钟K线没有amount，用0填充
    x_timestamp = df.iloc[-lookback:]['datetime']
    
    last_time = df.iloc[-1]['datetime']
    future_times = pd.date_range(start=last_time + timedelta(minutes=5), periods=pred_len, freq='5min')
    y_timestamp = pd.Series(future_times)
    
    print(f"📊 输入数据: {lookback}根5分钟K线 | 预测: {pred_len}根未来K线")
    
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.7,
        top_p=0.9,
        sample_count=3,
    )
    
    return pred_df, x_df


def main():
    if len(sys.argv) < 2:
        print("用法: python3 kronos_predict.py <股票代码>")
        print("示例: python3 kronos_predict.py sz.300696")
        sys.exit(1)
    
    stock_code = sys.argv[1]
    stock_name = ""
    
    print(f"\n🔮 KRONOS 短线猎手 - K线预测")
    print(f"=" * 50)
    
    # 步骤1：获取实时价格（新浪API）
    print(f"📥 获取 {stock_code} 实时价格（新浪API）...")
    rt = get_realtime_price(stock_code)
    if rt:
        stock_name = rt["name"]
        print(f"✅ 实时价格: {rt['current']:.2f} 元 | 昨收: {rt['prev_close']:.2f} | 今开: {rt['open']:.2f}")
        print(f"   今日最高: {rt['high']:.2f} | 今日最低: {rt['low']:.2f}")
    
    # 步骤2：获取5分钟K线（baostock）
    print(f"\n📥 获取 {stock_code} 5分钟K线数据（baostock）...")
    df = get_5min_kline(stock_code)
    
    if df is None or len(df) < 50:
        print("❌ 获取数据失败或数据不足")
        sys.exit(1)
    
    print(f"✅ 获取到 {len(df)} 根5分钟K线")
    print(f"   时间范围: {df.iloc[0]['datetime']} ~ {df.iloc[-1]['datetime']}")
    
    # 步骤3：加载KRONOS模型
    try:
        predictor = load_kronos_model()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 步骤4：执行预测
    print(f"\n🔮 正在调用 KRONOS 模型预测...")
    try:
        pred_df, x_df = run_prediction(predictor, df)
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 步骤5：分析结果（使用实时价格）
    if rt:
        original_price = rt["current"]
    else:
        original_price = float(x_df.iloc[-1]['close'])
    
    pred_close = pred_df['close'].values
    pred_high = pred_df['high'].values
    
    max_high_price = float(pred_high.max())
    max_high_idx = int(pred_high.argmax())
    final_close = float(pred_close[-1])
    final_return = (final_close - original_price) / original_price * 100
    max_return = (max_high_price - original_price) / original_price * 100
    limit_up = round(original_price * 1.1, 2)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"🔮 KRONOS K线预测结果 | {stock_name} {stock_code}")
    print(f"{'='*60}")
    
    print(f"\n📊 实时当前价: {original_price:.2f} 元")
    print(f"   涨停价: {limit_up:.2f} 元（+10%需 {limit_up-original_price:.2f}元）")
    print(f"📈 预测最高价: {max_high_price:.2f} 元（{max_return:+.2f}%）")
    print(f"   出现在第 {max_high_idx+1} 根K线（约 {(max_high_idx+1)*5} 分钟后）")
    print(f"📉 预测收盘: {final_close:.2f} 元（{final_return:+.2f}%）")
    
    print(f"\n预测K线详情：")
    print(f"{'K':>3} | {'开盘':>8} | {'最高':>8} | {'最低':>8} | {'收盘':>8} | {'涨跌':>8}")
    print("-" * 60)
    for i in range(min(24, len(pred_df))):
        row = pred_df.iloc[i]
        open_p = float(row['open'])
        high_p = float(row['high'])
        low_p = float(row['low'])
        close_p = float(row['close'])
        ret = (close_p - original_price) / original_price * 100
        marker = " ←最高" if i == max_high_idx else ""
        print(f"{i+1:3d} | {open_p:8.2f} | {high_p:8.2f} | {low_p:8.2f} | {close_p:8.2f} | {ret:+7.2f}%{marker}")
    
    # 收益计算
    invest = 30000
    shares = invest // original_price // 100 * 100
    limit_profit = shares * (limit_up - original_price)
    max_profit = shares * (max_high_price - original_price)
    close_profit = shares * (final_close - original_price)
    
    print(f"\n💰 收益预期（本金{invest}元）")
    print(f"   实时买入价: {original_price:.2f} | 可买股数: {shares}股")
    print(f"   涨停收益 (+10%): +{limit_profit:.0f} 元")
    print(f"   预测最高收益 ({max_return:+.1f}%): +{max_profit:.0f} 元")
    print(f"   预测收盘收益 ({final_return:+.1f}%): +{close_profit:.0f} 元")
    
    print(f"\n⚠️  免责声明：预测结果仅供学术研究，不构成投资建议")


if __name__ == "__main__":
    main()
