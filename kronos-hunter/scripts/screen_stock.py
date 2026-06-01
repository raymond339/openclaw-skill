#!/usr/bin/env python3
"""
KRONOS 短线猎手 - 主选股脚本
用法: python3 screen_stock.py <热点关键词>
示例: python3 screen_stock.py 军工

依赖: baostock, requests
安装: pip3 install baostock requests -q
"""

import sys
import os
import subprocess
import json

# 热点板块映射
HOTKEY_BOARD_MAP = {
    "AI": "AI算力",
    "人工智能": "AI算力",
    "GPT": "AI算力",
    "算力": "AI算力",
    "大模型": "AI算力",
    "芯片": "半导体",
    "半导体": "半导体",
    "光刻": "半导体",
    "新能源": "新能源汽车",
    "电动车": "新能源汽车",
    "锂电": "新能源汽车",
    "储能": "新能源汽车",
    "军工": "军工航天",
    "航天": "军工航天",
    "航空": "军工航天",
    "成飞": "军工航天",
    "券商": "非银金融",
    "牛市": "非银金融",
    "降准": "非银金融",
    "并购": "非银金融",
    "医疗": "医药",
    "创新药": "医药",
    "中药": "医药",
    "消费": "消费",
    "食品": "消费",
    "白酒": "消费",
    "房地产": "房地产链",
    "建材": "房地产链",
    "家居": "房地产链",
}

# 简化板块成分股
BOARD_STOCKS = {
    "AI算力": ["sh.600588", "sh.601012", "sz.300308", "sz.300274", "sh.600406"],
    "半导体": ["sz.002371", "sh.688981", "sz.300782", "sh.688396", "sz.688008"],
    "新能源汽车": ["sz.300750", "sh.601238", "sz.002594", "sh.600733", "sz.300124"],
    "军工航天": ["sz.302132", "sh.600893", "sh.600038", "sz.300696", "sh.688185"],
    "非银金融": ["sh.600109", "sh.601066", "sh.601601", "sz.000712", "sh.600030"],
    "医药": ["sh.600276", "sz.000538", "sh.601607", "sz.300760", "sh.688180"],
    "消费": ["sz.000858", "sh.600519", "sh.603288", "sz.002304", "sh.600887"],
    "房地产链": ["sz.000002", "sh.600048", "sh.601155", "sz.002146", "sh.600383"],
}


def get_realtime_price(stock_code):
    """从新浪API获取实时价格数据"""
    try:
        # 转换代码格式: sh.600588 -> sh600588, sz.302132 -> sz302132
        code = stock_code.replace(".", "")
        url = f"https://hq.sinajs.cn/list={code}"
        
        result = subprocess.run(
            ["curl", "-s", "--max-time", "5", url, "-H", "Referer: https://finance.sina.com.cn"],
            capture_output=True, timeout=10
        )
        
        # 新浪返回GBK编码，需要转换
        try:
            output = result.stdout.decode("gbk").strip()
        except UnicodeDecodeError:
            output = result.stdout.decode("utf-8", errors="ignore").strip()
        if not output or "none" in output.lower():
            return None
        
        # 解析: var hq_str_sh600588="名称,当前价,昨收,今开,最高,最低,..."
        # 新浪API字段: 0=名称, 1=当前价(有时不准), 2=昨收, 3=今开, 4=最高, 5=最低, 7=卖一(实时参考价), 8=成交量, 9=成交额
        # 注: 字段[1]在交易时段有时返回滞后价格，字段[7]卖一价格更可靠
        start = output.find('"')
        end = output.rfind('"')
        if start == -1 or end == -1:
            return None
        
        data = output[start+1:end].split(",")
        if len(data) < 10:
            return None
        
        # 用字段[7](卖一)作为当前参考价，更可靠
        current_price = float(data[7]) if data[7] and data[7] != '0' else float(data[1])
        
        return {
            "name": data[0],
            "current": current_price,
            "prev_close": float(data[2]) if data[2] else 0,
            "open": float(data[3]) if data[3] else 0,
            "high": float(data[4]) if data[4] else 0,
            "low": float(data[5]) if data[5] else 0,
            "volume": int(data[8]) if data[8] else 0,
            "amount": float(data[9]) if data[9] else 0,
        }
    except Exception as e:
        print(f"  获取 {stock_code} 实时价格失败: {e}")
        return None


def get_baostock_kline(stock_code, days=10):
    """从baostock获取近日期K线数据计算技术指标"""
    try:
        import baostock as bs
        lg = bs.login()
        
        from datetime import datetime, timedelta
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if len(data_list) < 5:
            return None
        
        import pandas as pd
        df = pd.DataFrame(data_list, columns=rs.fields)
        for col in ["open", "high", "low", "close", "volume", "turn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 计算MA和动能
        df = df.sort_values("date").reset_index(drop=True)
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["mom5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100
        
        # 成交量平均
        df["vol_ma5"] = df["volume"].rolling(5).mean()
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        return {
            "MA5": float(latest["MA5"]) if not pd.isna(latest["MA5"]) else 0,
            "MA10": float(latest["MA10"]) if not pd.isna(latest["MA10"]) else 0,
            "mom5": float(latest["mom5"]) if not pd.isna(latest["mom5"]) else 0,
            "vol_ratio": float(latest["volume"]) / float(latest["vol_ma5"]) if latest["vol_ma5"] > 0 else 0,
            "turn_rate": float(latest["turn"]) if not pd.isna(latest["turn"]) else 0,
        }
    except Exception as e:
        print(f"  baostock数据获取失败: {e}")
        return None


def calc_limit_up(prev_close):
    """计算涨停价（A股主板10%，科创板/创业板20%）"""
    return round(prev_close * 1.1, 2)


def score_stock(current, prev_close, high, low, kline_data):
    """综合评分：涨停概率"""
    limit_up = calc_limit_up(prev_close)
    dist_to_limit = (limit_up - current) / limit_up * 100 if limit_up > 0 else 100
    
    # 动量分（30%）
    mom5 = kline_data.get("mom5", 0) if kline_data else 0
    mom_score = min(max(mom5 / 8 * 100, 0), 100) * 0.3
    
    # 量比分（20%）
    vol_ratio = kline_data.get("vol_ratio", 0) if kline_data else 0
    vol_score = min(vol_ratio / 2 * 100, 100) * 0.2 if vol_ratio > 0 else 0
    
    # 换手率分（15%）
    turn = kline_data.get("turn_rate", 0) if kline_data else 0
    turn_score = min(turn / 5 * 100, 100) * 0.15 if turn > 0 else 0
    
    # 距涨停分（35%）
    dist_score = max(0, (10 - dist_to_limit) / 10 * 100) * 0.35
    
    return round(mom_score + vol_score + turn_score + dist_score, 1), round(dist_to_limit, 2)


def screen_board(keyword):
    """扫描板块，返回评分最高的股票"""
    board = HOTKEY_BOARD_MAP.get(keyword)
    if not board:
        return None, f"未找到板块: {keyword}，请尝试：AI/芯片/新能源/军工/券商/医疗/消费/房地产"
    
    stocks = BOARD_STOCKS.get(board, [])
    if not stocks:
        return None, f"板块 {board} 暂无成分股数据"
    
    results = []
    
    print(f"\n🎯 板块: {board} | 关键词: {keyword}")
    print("=" * 60)
    
    for code in stocks:
        # 获取实时价格
        rt = get_realtime_price(code)
        if not rt or rt["current"] == 0:
            print(f"  {code}: 实时价格获取失败")
            continue
        
        # 获取历史K线数据计算指标
        kline = get_baostock_kline(code)
        
        # 计算涨停概率评分
        score, dist = score_stock(
            current=rt["current"],
            prev_close=rt["prev_close"],
            high=rt["high"],
            low=rt["low"],
            kline_data=kline
        )
        
        results.append({
            "code": code,
            "name": rt["name"],
            "current": rt["current"],
            "prev_close": rt["prev_close"],
            "open": rt["open"],
            "high": rt["high"],
            "low": rt["low"],
            "limit_up": calc_limit_up(rt["prev_close"]),
            "dist_to_limit": dist,
            "mom5": kline.get("mom5", 0) if kline else 0,
            "vol_ratio": kline.get("vol_ratio", 0) if kline else 0,
            "turn_rate": kline.get("turn_rate", 0) if kline else 0,
            "score": score,
        })
        
        print(f"  {code} {rt['name']}: 现价{rt['current']:.2f} | 距涨停{dist:.2f}% | 5日动能{(kline.get('mom5', 0) if kline else 0):+.1f}% | 评分{score}")
    
    if not results:
        return None, "所有股票价格获取失败"
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3], board


def main():
    if len(sys.argv) < 2:
        print("用法: python3 screen_stock.py <热点关键词>")
        print("示例: python3 screen_stock.py 军工")
        sys.exit(1)
    
    keyword = sys.argv[1]
    results, board = screen_board(keyword)
    
    if results is None:
        print(f"错误: {board}")
        sys.exit(1)
    
    print(f"\n🏆 Top 3 推荐:")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n#{i} {r['name']} ({r['code']})")
        print(f"   当前价: {r['current']:.2f} | 涨停价: {r['limit_up']:.2f} | 距涨停: {r['dist_to_limit']:.2f}%")
        print(f"   5日动能: {r['mom5']:+.2f}% | 量比: {r['vol_ratio']:.2f} | 换手率: {r['turn_rate']:.2f}%")
        print(f"   涨停概率评分: {r['score']:.1f}/100")


if __name__ == "__main__":
    main()
