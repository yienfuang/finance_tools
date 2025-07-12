import datetime as dt
import numpy as np
import pandas as pd
import pytz
import time
import talib

# stand up Binance API client instance
import os
from binance.client import Client

# plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multiprocessing import Pool
from contextlib import closing

import logging

df = pd.read_parquet("D:/simTrades_trackSL.parquet")
mc = pd.read_parquet("D:/market_capitalisation_20250609.parquet")
df = df[df.pair.isin(mc.pair)].reset_index(drop=True)
df = df.merge(mc, how="left", on="pair")

dfList = [g for _,g in df.groupby(["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"])]

def getSelectiveTrades(df, n=10):
    return df.nsmallest(n, "rank")

logger = logging.getLogger()
handler = logging.FileHandler("D:/simNoSL.log")
logger.addHandler(handler)

def simNoSL(subset):

    a = subset.MAForBuy.iloc[0]
    b = subset.abovePCForBuy.iloc[0]
    c = subset.MAForSell.iloc[0]
    d = subset.belowPCForSell.iloc[0]

    startingMsg = f"Simulating MAForBuy:{a} abovePCForBuy:{b} MAForSell:{c} belowPCForSell:{d}"
    print(startingMsg)
    logger.error(startingMsg)

    tradeTime = subset.buyTime.min()
    logger.error(f"First buy time: {tradeTime}")

    positions = subset[subset.buyTime==tradeTime].reset_index(drop=True)
    logger.error(f"{len(positions)} trades available for first buy")

    positions = getSelectiveTrades(positions, n=10)
    logger.error(f"{len(positions)} trades selected for first buy")

    positions[["entrySizeNoSL", "exitSizeNoSL"]] = [.1, np.nan]
    logger.error("First buy entry and exit sizes initialised")
    
    portfolioSize = pd.DataFrame({"date": tradeTime, "portNoSL": 1}, index=[0])
    logger.error("Portfolio size tracker initialised")

    while tradeTime <= subset.buyTime.max():
        
        nextBuy = subset[subset.buyTime>tradeTime].buyTime.min()
        nextSell = positions[positions.exitSizeNoSL.isna()].sellTime.min()
        logger.error(f"Next buy time: {nextBuy}, next sell time: {nextSell}")

        tradeIsBuy = True if positions.exitSizeNoSL.isna().sum()==0 else nextBuy <= nextSell
        logger.error(f"Therefore, tradeIsBuy is {tradeIsBuy}")

        tradeTime = nextBuy if tradeIsBuy else nextSell
        logger.error(f"Trade time set to: {tradeTime}")

        if tradeIsBuy:
            
            posToAdd = subset[subset.buyTime==tradeTime].reset_index(drop=True)
            logger.error(f"Possible {len(posToAdd)} trades for buy at {tradeTime}")

            noTradesToAdd = max(0, 10 - positions.exitSizeNoSL.isna().sum())
            logger.error(f"Possible {noTradesToAdd} trades to add to positions because of current open positions")

            posToAdd = getSelectiveTrades(posToAdd, n=noTradesToAdd)
            logger.error(f"Selected {len(posToAdd)} trades to add to positions")

            entrySize = .1 * portfolioSize.portNoSL.iloc[-1]
            posToAdd["entrySizeNoSL"] = entrySize
            logger.error(f"New trade(s) entry size is {entrySize}")

            positions = pd.concat([positions, posToAdd], ignore_index=True)
            logger.error(f"Positions now have {len(positions)} entries after adding new trades")

        else:
            
            positions.loc[positions.sellTime==tradeTime, "exitSizeNoSL"] = positions.entrySizeNoSL*(1+positions.profit)
            noClosedTrades = (positions.sellTime == tradeTime).sum()
            logger.error(f"Closed {noClosedTrades} trades")

            newSize = positions[positions.sellTime==tradeTime].exitSizeNoSL.sum()
            portfolioSize.loc[len(portfolioSize), ["date", "portNoSL"]] = [tradeTime, portfolioSize.portNoSL.iloc[-1] * (1 - noClosedTrades / 10) + newSize]
            logger.error(f"New portfolio size after closing trades: {portfolioSize.portNoSL.iloc[-1]}")
            
            if nextBuy == nextSell:

                logger.error("Buy time same as sell time. Process the entries")

                posToAdd = subset[subset.buyTime==tradeTime].reset_index(drop=True)
                logger.error(f"Possible {len(posToAdd)} trades for buy at {tradeTime}")

                noTradesToAdd = max(0, 10 - positions.exitSizeNoSL.isna().sum())
                logger.error(f"Possible {noTradesToAdd} trades to add to positions because of current open positions")

                posToAdd = getSelectiveTrades(posToAdd, n=noTradesToAdd)
                logger.error(f"Selected {len(posToAdd)} trades to add to positions")

                entrySize = .1 * portfolioSize.portNoSL.iloc[-1]
                posToAdd["entrySizeNoSL"] = entrySize
                logger.error(f"New trade(s) entry size is {entrySize}")

                positions = pd.concat([positions, posToAdd], ignore_index=True)
                logger.error(f"Positions now have {len(positions)} entries after adding new trades")

    portfolioSize[["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"]] = a, b, c, d
    positions.to_parquet(f"D:/simNoSL_{a}_{b}_{c}_{d}.parquet")
    savedPositionsMesg = f"Saved positions for MAForBuy:{a} abovePCForBuy:{b} MAForSell:{c} belowPCForSell:{d}"
    print(savedPositionsMesg)
    logger.error(savedPositionsMesg)
    portfolioSize.to_parquet(f"D:/portfolioSizeNoSL_{a}_{b}_{c}_{d}.parquet")
    savedPortfolioMesg = f"Saved portfolio size for MAForBuy:{a} abovePCForBuy:{b} MAForSell:{c} belowPCForSell:{d}"
    print(savedPortfolioMesg)
    logger.error(savedPortfolioMesg)

if __name__ == "__main__":
    with closing(Pool(12)) as p:
        p.map(simNoSL, dfList)

# mc = pd.read_excel("D:/market_capitalisation.xlsx", sheet_name="2025-06-09")
# mc.coin = mc.coin.shift(-2)
# mc = mc[mc["rank"].notna()].reset_index(drop=True)
# mc = mc.groupby("coin")["rank"].nsmallest(1).reset_index()[["coin", "rank"]].sort_values("rank", ascending=True).reset_index(drop=True)
# mc["pair"] = mc.coin + "USDT"
# mc[["rank", "pair"]].to_parquet("D:/market_capitalisation_20250609.parquet")

# def getSelectiveTrades(df, n=10):
#     return df.nsmallest(n, "rank")

# positionList = []
# portfolioSizeList = []

# @njit(parallel=True)
# def simNoSL(df):

#     def getSelectiveTrades(df, n=10):
#         return df.nsmallest(n, "rank")

#     for a in prange(201):
#         for b in prange(21):
#             for c in prange(201):
#                 for d in prange(21):
#                     print(f"Simulating {a} {b} {c} {d}")
#                     subset = df[(df.MAForBuy==a) & (df.abovePCForBuy==b) & (df.MAForSell==c) & (df.belowPCForSell==d)].reset_index(drop=True)
#                     if len(subset) == 0:
#                         continue
#                     tradeTime = subset.buyTime.min()
#                     positions = subset[subset.buyTime==tradeTime].reset_index(drop=True)
#                     positions = getSelectiveTrades(positions, n=10)
#                     for col in ["entrySizeNoSL"]:
#                         positions[col] = .1
#                     for col in ["exitSizeNoSL"]:
#                         positions[col] = np.nan
#                     portfolioSize = pd.DataFrame({"date": tradeTime, "portNoSL": 1}, index=[0])
#                     while tradeTime <= subset.buyTime.max():
#                         nextBuy = subset[subset.buyTime>tradeTime].buyTime.min()
#                         nextSell = positions.sellTime.min()
#                         tradeIsBuy = nextBuy <= nextSell
#                         tradeTime = nextBuy if tradeIsBuy else nextSell
#                         if tradeIsBuy:
#                             posToAdd = subset[subset.buyTime==tradeTime].reset_index(drop=True)
#                             noTradesToAdd = max(0, 10 - positions.exitSizeNoSL.isna().sum())
#                             posToAdd = getSelectiveTrades(posToAdd, n=noTradesToAdd)
#                             positions = pd.concat([positions, posToAdd], ignore_index=True)
#                         else:
#                             positions.loc[positions.sellTime==tradeTime, "exitSizeNoSL"] = positions.entrySizeNoSL*(1+positions.profit)
#                             noClosedTrades = (positions.sellTime == tradeTime).sum()
#                             newSize = positions[positions.sellTime==tradeTime].exitSizeNoSL.sum()
#                             portfolioSize.loc[len(portfolioSize), ["date", "portNoSL"]] = [tradeTime, portfolioSize.portNoSL.iloc[-1] * (1 - noClosedTrades / 10) + newSize]
                            
#                             if nextBuy == nextSell:
#                                 posToAdd = subset[subset.buyTime==tradeTime].reset_index(drop=True)
#                                 noTradesToAdd = max(0, 10 - positions.exitSizeNoSL.isna().sum())
#                                 posToAdd = getSelectiveTrades(posToAdd, n=noTradesToAdd)
#                                 positions = pd.concat([positions, posToAdd], ignore_index=True)
#                     positions[["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"]] = a, b, c, d
#                     portfolioSize[["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"]] = a, b, c, d
#                     positions.to_parquet(f"D:/simNoSL_{a}_{b}_{c}_{d}.parquet")
#                     portfolioSize.to_parquet(f"D:/portfolioSizeNoSL_{a}_{b}_{c}_{d}.parquet")
#                     # positionList.append(positions)
#                     # portfolioSizeList.append(portfolioSize)

# simNoSL(df)

# for f in os.listdir("D:/binanceOHLC/"):
#     if f.endswith("USDT.parquet"):
#         try:
#             price = pd.read_parquet(f"D:/binanceOHLC/{f}")
#         except Exception as e:
#             print(f"Error reading {f}: {e}")
#             continue
#         pair = f.replace(".parquet", "")
#         for i in df.index[df.pair==pair]:
#             df.loc[i,"localMinPrice"] = price[price.openTime.between(df.loc[i,"buyTime"],df.loc[i,"sellTime"],inclusive="right")].low.min()

# for sl in [.05, .1, .15, .2]:
#     slStr = str(int(sl*100)).zfill(2)
#     df[f"SL{slStr}Profit"] = np.where((df.buyPrice - df.localMinPrice)/df.buyPrice>=sl, -sl, df.profit)


# dfList = []
# for pq in os.listdir("D:/binanceOHLC/"):
#     if pq.endswith("USDT.parquet"):
#         try:
#             btc = pd.read_parquet(f"D:/binanceOHLC/{pq}")
#         except Exception as e:
#             print(f"Error reading {pq}: {e}")
#             continue
#     for r in [3,7,14,30,200]:
#         maCol = f"SMA{r}"
#         btc[maCol] = talib.SMA(btc.close, timeperiod=r)
#     for ad in [3,7,14,30,200]:
#         for apc in [5,10,20]:
#             amaCol = f"SMA{ad}"
#             aboveCol = f"is{apc}pcAbove{amaCol}"
#             btc[aboveCol] = (btc.close > btc[amaCol]*(1+apc/100)).astype(int)*btc.close
#             for bd in [3,7,14,30,200]:
#                 for bpc in [5,10,20]:
#                     bmaCol = f"SMA{bd}"
#                     belowCol = f"is{bpc}pcBelow{bmaCol}"
                    
#                     buys = btc.loc[(btc[aboveCol]>0) & (btc[bmaCol].notna()), ["openTime", "close"]].reset_index(drop=True)
#                     if len(buys) == 0:
#                         print(f"Buying {apc}% above {amaCol} and selling {bpc}% below {bmaCol} gives no signal")
#                         continue
#                     buys.columns = ["buyTime", "buyPrice"]
#                     buys["gapCheck"] = buys.buyTime.diff()

#                     btc[belowCol] = (btc.close < btc[bmaCol]*(1-bpc/100)).astype(int)*btc.close
#                     sells = btc.loc[btc[belowCol]>0, ["openTime", "close"]].reset_index(drop=True)
#                     sells.rename({"close": "sellPrice"}, axis=1, inplace=True)
#                     for r in range(len(buys)):
#                         buyTime = buys.buyTime[r]
#                         sellTime = sells.openTime[sells.openTime>buyTime].min()
#                         if not pd.isna(sellTime):
#                             buys.loc[r, "sellPrice"] = btc.loc[btc.openTime==sellTime, "close"].values[0]
#                             buys.loc[r, "sellTime"] = sellTime
#                             buys.loc[r, "profit"] = buys.sellPrice[r]/buys.buyPrice[r] - 1
#                         else:
#                             buys.loc[r, "sellPrice"] = np.nan
#                             buys.loc[r, "sellTime"] = np.nan
#                             buys.loc[r, "profit"] = np.nan

#                     buys["pair"] = pq.replace(".parquet", "")
#                     buys["MAForBuy"] = ad
#                     buys["abovePCForBuy"] = apc
#                     buys["MAForSell"] = bd
#                     buys["belowPCForSell"] = bpc

#                     dfList.append(buys)
# allTrades = pd.concat(dfList, ignore_index=True)

# for sl in [.05, .1, .15, .2]:
#     slStr = str(int(sl*100)).zfill(2)
#     allTrades[f"SL{slStr}Profit"] = allTrades.profit.clip(lower=-sl)

# allTrades = df.copy()

# oneBuyATime = allTrades[(allTrades.profit.notna()) & (allTrades.gapCheck>dt.timedelta(days=1))].reset_index(drop=True)
# oneBuyATime.groupby(["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"])["profit"].sum().sort_values(ascending=False)
# oneBuyATime.groupby(["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"])["SL20Profit"].sum().sort_values(ascending=False)

# perf = pd.read_parquet("D:/binanceOHLC/performance.parquet")

# dfList = []
# for pq in os.listdir("D:/binanceOHLC/"):
#     if pq.endswith(".parquet"):
#         try:
#             btc = pd.read_parquet(f"D:/binanceOHLC/{pq}")
#         except Exception as e:
#             print(f"Error reading {pq}: {e}")
#             continue
#     for r in [3,7,14,30,200]:
#         maCol = f"SMA{r}"
#         btc[maCol] = talib.SMA(btc.close, timeperiod=r)
#     adList = []
#     apcList = []
#     bdList = []
#     bpcList = []
#     buyEverySignalList = []
#     buyEverySignalNoTrades = []
#     oneTradeOnAtATimeList = []
#     oneTradeOnAtATimeNoTrades = []
#     for ad in [3,7,14,30,200]:
#         for apc in [5,10,20]:
#             amaCol = f"SMA{ad}"
#             aboveCol = f"is{apc}pcAbove{amaCol}"
#             btc[aboveCol] = (btc.close > btc[amaCol]*(1+apc/100)).astype(int)*btc.close
#             for bd in [3,7,14,30,200]:
#                 for bpc in [5,10,20]:
#                     bmaCol = f"SMA{bd}"
#                     belowCol = f"is{bpc}pcBelow{bmaCol}"
                    
#                     buys = btc.loc[(btc[aboveCol]>0) & (btc[bmaCol].notna()), ["openTime", "close"]].reset_index(drop=True)
#                     if len(buys) == 0:
#                         print(f"Buying {apc}% above {amaCol} and selling {bpc}% below {bmaCol} gives no signal")
#                         continue
#                     buys.columns = ["buyTime", "buyPrice"]
#                     buys["gapCheck"] = buys.buyTime.diff()

#                     btc[belowCol] = (btc.close < btc[bmaCol]*(1-bpc/100)).astype(int)*btc.close
#                     sells = btc.loc[btc[belowCol]>0, ["openTime", "close"]].reset_index(drop=True)
#                     sells.rename({"close": "sellPrice"}, axis=1, inplace=True)
#                     for r in range(len(buys)):
#                         buyTime = buys.buyTime[r]
#                         sellTime = sells.openTime[sells.openTime>buyTime].min()
#                         if not pd.isna(sellTime):
#                             buys.loc[r, "sellPrice"] = btc.loc[btc.openTime==sellTime, "close"].values[0]
#                             buys.loc[r, "sellTime"] = sellTime
#                             buys.loc[r, "profit"] = buys.sellPrice[r]/buys.buyPrice[r] - 1
#                         else:
#                             buys.loc[r, "sellPrice"] = np.nan
#                             buys.loc[r, "sellTime"] = np.nan
#                             buys.loc[r, "profit"] = np.nan
#                     adList.append(ad)
#                     apcList.append(apc)
#                     bdList.append(bd)
#                     bpcList.append(bpc)
#                     buyEverySignalList.append(buys[buys.profit.notna()].profit.sum())
#                     buyEverySignalNoTrades.append(len(buys[buys.profit.notna()]))
#                     oneTradeOnAtATimeList.append(buys[(buys.profit.notna()) & (buys.gapCheck>dt.timedelta(days=1))].profit.sum())
#                     oneTradeOnAtATimeNoTrades.append(len(buys[(buys.profit.notna()) & (buys.gapCheck>dt.timedelta(days=1))]))
#     performance = pd.DataFrame({"MAForBuy": adList, "abovePCForBuy": apcList,
#                                 "MAForSell": bdList, "belowPCForSell": bpcList,
#                                 "performanceBuyEverySignal": buyEverySignalList, "noTradesBuyEverySignal": buyEverySignalNoTrades,
#                                 "performanceOneTradeAtATime": oneTradeOnAtATimeList, "noTradesOneTradeAtATime": oneTradeOnAtATimeNoTrades})
#     performance["pair"] = pq.replace(".parquet", "")
#     dfList.append(performance)
# performance = pd.concat(dfList, ignore_index=True)
# performance.to_parquet("D:/binanceOHLC/performance.parquet")
# perf.sort_values("performanceOneTradeAtATime", ascending=False)
# perf.groupby(["MAForBuy", "abovePCForBuy", "MAForSell", "belowPCForSell"])["performanceOneTradeAtATime"].sum().sort_values(ascending=False)

# df = pd.read_parquet("D:/binanceOHLC/DOGEUSDT.parquet")
# px.line(df, x="openTime", y="close", title="DOGEUSDT").show()

# plotData = btc.melt(id_vars=["openTime"], value_vars=["close"
# "", "is5pcAbove3DMA", "is5pcBelow3DMA"]+[f"SMA{r}" for r in [3,7,14,30,200]])
# fig = px.line(plotData, x="openTime", y="value", color="variable", title="BTCUSDT SMA")
# fig.show()

# talib.SMA(btc.close, timeperiod=20)
# talib.ATR(btc.close, btc.high, btc.low, timeperiod=14)
# talib.CDLEVENINGDOJISTAR(btc.open, btc.high, btc.low, btc.close).unique()

# api_key = os.environ.get("binance_api")
# api_secret = os.environ.get("binance_secret")
# bClient = Client(api_key=api_key, api_secret=api_secret)

# allPairs = pd.DataFrame(bClient.get_all_tickers())
# allPairs = allPairs[allPairs.symbol.str.endswith("USDT")].reset_index(drop=True)

# for p in allPairs.symbol:
#     ohlc = []
#     for y in range(2017, 2026):
#         startDay = dt.datetime(y, 1, 1)
#         endDay = dt.datetime(y, 12, 31)
#         try:
#             ohlc += bClient.get_historical_klines(symbol=p, interval=bClient.KLINE_INTERVAL_1DAY, start_str=str(startDay), end_str=str(endDay))
#             print(f"Got {p} {startDay} to {endDay} data")
#         except Exception as e:
#             print(f"Error: {e}")
#     ohlc = pd.DataFrame(ohlc)
#     ohlc.columns = ["openTime", "open", "high", "low", "close", "volume", "closeTime", "quoteAssetVol", "numOfTrades", "takerBuyBaseAssetVol", "takerBuyQuoteAssetVol", "ignore"]
#     for c in ["openTime", "closeTime"]:
#         ohlc[c] = pd.to_datetime(ohlc[c], unit="ms")
#     for c in ["open", "high", "low", "close", "volume", "quoteAssetVol", "takerBuyBaseAssetVol", "takerBuyQuoteAssetVol"]:
#         ohlc[c] = ohlc[c].astype(float)
#     ohlc.to_parquet(f"D:/binanceOHLC/{p}.parquet")




# # visualy check the data
# def plotOHLC(pair:str, startTime=None, endTime=None):
#     """
#     Plot OHLC data

#     Parameters
#     ----------
#     pair      : Coin pair as listed in Binance
#     startTime  : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
#     endTime   : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
#     """
#     filename = f"D:/binanceOHLC/{pair}.parquet"
#     if not os.path.exists(filename):
#         print(f"File {filename} does not exist")
#         return
#     ohlc = pd.read_parquet(filename)

#     if startTime:
#         ohlc = ohlc[ohlc.openTime>=startTime].copy()
#     if endTime:
#         ohlc = ohlc[ohlc.openTime<=endTime].copy()
    
#     fig = go.Figure(data=go.Ohlc(x=ohlc.openTime, open=ohlc.open, high=ohlc.high, low=ohlc.low, close=ohlc.close))
#     fig.show()

# plotOHLC("DOTUSDT")


# class BinanceClient:
#     """
#     Instantiate a Binance Client

#     Parameters
#     ----------
#     api_key    : Name of environment variable that stores Binance API key
#     api_secret : Name of environment variable that stores Binance API secret
#     """
#     def __init__(self, api_key:str="binance_api", api_secret:str="binance_secret"):
#         api_key = os.environ.get(api_key)
#         api_secret = os.environ.get(api_secret)
#         self.client = Client(api_key=api_key, api_secret=api_secret)

#     def get_1m_price(self, pair:str, time, timezone:str="UTC", priceType:str="o"):
#         """
#         Get 1-minute price

#         Parameters
#         ----------
#         pair      : Coin pair as listed in Binance
#         time      : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
#         timezone  : Timezone in str e.g. 'UTC', 'Australia/Melbourne', 'Australia/Brisbane'
#         priceType : 'o' for open; 'h' for high; 'l' for low; 'c' for close; others to return all
#         """

#         # round timestamp down to nearest minute
#         time = pd.Timestamp(time, tz=timezone).round(freq="T")

#         # Binance takes UNIX timestamp in milliseconds
#         time = 1000*time.timestamp()

#         # OHLC price
#         ohlc = self.client.get_historical_klines(symbol=pair, interval=self.client.KLINE_INTERVAL_1MINUTE,
#                                                  start_str=str(time), end_str=str(time + 1))

#         # price to return
#         if priceType=="o":
#             priceType = 1
#         elif priceType=="h":
#             priceType = 2
#         elif priceType=="l":
#             priceType = 3
#         elif priceType=="c":
#             priceType = 4
#         else:
#             return ohlc
        
#         return float(ohlc[0][priceType])
    
#     def get_aud_rate(self, coin:str, time, timezone:str="UTC"):
#         """
#         Get coin rate in AUD

#         Parameters
#         ----------
#         coin     : Coin name
#         time     : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
#         timezone : Timezone in str e.g. 'UTC', 'Australia/Melbourne', 'Australia/Brisbane'
#         """
#         timeStr = f"{str(time)} {timezone}"
#         if coin in ["BUSD", "USDT"]: # BUSD and USDT are the base in Binance
#             pair = f"AUD{coin}"
#             try:
#                 return 1/self.get_1m_price(pair=pair, time=time, timezone=timezone)
#             except:
#                 print(f"{pair} price at {timeStr} not found")
#         elif coin == "BETH": # BETH starts with only BETH/ETH pair
#             try:
#                 return self.get_1m_price(pair="BETHETH", time=time, timezone=timezone) * self.get_1m_price(pair="ETHUSDT", time=time, timezone=timezone)
#             except:
#                 print(f"BETHETH or/and ETHAUD price at {timeStr} not found")
#         else: # for other coins, try all possible bases i.e. AUD, USDT, and BUSD
#             try:
#                 pair = f"{coin}AUD"
#                 return self.get_1m_price(pair=pair, time=time, timezone=timezone)
#             except:
#                 print(f"{pair} price at {timeStr} not found")
#                 try:
#                     pair = f"{coin}USDT"
#                     return self.get_1m_price(pair=pair, time=time, timezone=timezone) # / self.get_1m_price(pair="AUDUSDT", time=time, timezone=timezone)
#                 except:
#                     print(f"{pair} price at {timeStr} not found")
#                     try:
#                         pair = f"{coin}BUSD"
#                         return self.get_1m_price(pair=pair, time=time, timezone=timezone) # / self.get_1m_price(pair="AUDBUSD", time=time, timezone=timezone)
#                     except:
#                         print(f"{pair} price at {timeStr} not found")

#     def get_trades(self, startTime=None, endTime=None, timezone:str="UTC", pair=None):
#         """
#         Get trades within time window. If startTime and endTime are not entered, all trades will be returned

#         Parameters
#         ----------
#         startTime : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
#         endTime   : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
#         timezone  : Timezone in str e.g. 'UTC', 'Australia/Melbourne', 'Australia/Brisbane'
#         pair      : Particular coin pair(s) to get trades for. If None, return all coin pairs' trades
#         """

#         # convert user-input times into Binance UNIX timestamp (ms)
#         if startTime:
#             startTime = int(1000*pd.Timestamp(startTime, tz=timezone).timestamp())
#         if endTime:
#             endTime = int(1000*pd.Timestamp(endTime, tz=timezone).timestamp())
        
#         # get all coin pairs
#         if type(pair) == str:
#             pair = [pair]
#         allPairs = pd.DataFrame({"symbol": pair}, index=np.arange(len(pair))) if pair else pd.DataFrame(self.client.get_all_tickers())

#         # go through all coin pairs and get my trades
#         trades = []
#         for r in range(len(allPairs)):
#             pair = allPairs.symbol[r]

#             try:
#                 trades.extend(self.client.get_my_trades(symbol=pair, limit=1000, startTime=startTime, endTime=endTime))
#                 print(f"Extended {pair} trades")
#             except Exception as e:
#                 print(f"Tried extending {pair} trades with error {e}")

#             # 1-sec wait time to avoid API rate limit
#             time.sleep(1)

#         trades = pd.DataFrame(trades)

#         # trades = trades[["symbol", "price", "qty", "quoteQty", "commission", "commissionAsset", "time", "isBuyer"]].copy()
#         # trades.rename({"symbol": "pair", "quoteQty": "baseQty"}, axis=1, inplace=True)
#         # trades["UTC"] = pd.to_datetime(trades.time, unit="ms")

#         # for r in range(len(trades)):
#         #     pair = trades.pair[r]
#         #     price = float(trades.price[r])
#         #     ts = trades.UTC[r]
#         #     if pair.endswith("AUD"):
#         #         trades.loc[r, "priceAUD"] = price
#         #     elif pair.startswith("AUD"):
#         #         trades.loc[r, "priceAUD"] = 1 / price
#         #     elif pair.endswith("BUSD"):
#         #         trades.loc[r, "priceAUD"] = price / self.get_1m_price("AUDBUSD", ts)
#         #     elif pair.endswith("USDT"):
#         #         trades.loc[r, "priceAUD"] = price / self.get_1m_price("AUDUSDT", ts)

#         return trades

# binanceClient = BinanceClient()



# trades = binanceClient.get_trades()
# trades = trades[["symbol", "price", "qty", "quoteQty", "commission", "commissionAsset", "time", "isBuyer"]].copy()
# trades.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2024/binance_trades.parquet")
# trades.rename({"symbol": "pair", "quoteQty": "baseQty"}, axis=1, inplace=True)
# trades["UTC"] = pd.to_datetime(trades.time, unit="ms")
# trades = trades[trades.UTC.between(dt.datetime(2023,6,30,14),dt.datetime(2024,6,30,14))].sort_values("UTC", ignore_index=True)
# trades.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2024/binance_trades_fy24.parquet")

# trades["nearesetUTC"] = trades.UTC.dt.round(freq="T")
# trades = trades.groupby(["nearesetUTC", "pair", "price", "commissionAsset", "isBuyer"])["qty", "baseQty", "commission"].sum().reset_index()


# allTrades = pd.read_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/binance_trades.parquet")

# # get Binance interest and staking rewards
# startTime = dt.datetime(2023,6,30,14)
# histInt = []
# while startTime < dt.datetime(2024,6,30,14):
#     endTime = startTime + dt.timedelta(days=1)
#     startUnixTime = int(1000 * dt.datetime.timestamp(startTime))
#     endUnixTime = int(1000 * dt.datetime.timestamp(endTime))
#     histInt.extend(binanceClient.client.get_asset_dividend_history(startTime=startUnixTime, endTime=endUnixTime, limit=500)["rows"])
#     startTime += dt.timedelta(days=1)
# histInt = pd.DataFrame(histInt)
# histInt["UTC"] = pd.to_datetime(histInt.divTime, unit="ms")
# histInt["nearestUTC"] = histInt.UTC.dt.round(freq="T")
# histInt.amount = histInt.amount.astype(float)
# histInt = histInt.groupby(["nearestUTC", "asset"])["amount"].sum().reset_index()

# for r in range(len(histInt)):
#     coin = "ETH" if histInt.asset[r] == "BETH" else histInt.asset[r]
#     time = histInt.nearestUTC[r]
#     try:
#         histInt.loc[r, "AUDPrice"] = binanceClient.get_aud_rate(coin=coin, time=time)
#     except Exception as e:
#         print(f"can't find price for {coin} at {time}, with error: {e}")
# histInt.loc[histInt.asset=="WOO", "AUDPrice"] = .3
# histInt.loc[(histInt.asset=="HNT") & (histInt.nearestUTC.dt.month.between(7,7)), "AUDPrice"] = 13.6
# histInt.loc[(histInt.asset=="HNT") & (histInt.nearestUTC.dt.month.between(8,8)), "AUDPrice"] = 10
# histInt.loc[(histInt.asset=="HNT") & (histInt.nearestUTC.dt.month.between(9,10)), "AUDPrice"] = 7.45
# histInt.loc[(histInt.asset=="HNT") & (histInt.nearestUTC>=dt.datetime(2022,11,1)), "AUDPrice"] = 3.5
# histInt["AUD"] = histInt.amount * histInt.AUDPrice
# """
# Price assumptions for coins that cannot be retrieved from swyftx api
# WOO price throughout FY23 assumed to be 0.2*1.5 AUD = 0.3AUD
# """
# histInt.AUD.sum()*len(histInt)/8058
# histInt.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/binance_interests.parquet")


# # get Binance withdrawal history
# withdraws = []
# startTime = dt.datetime(2021,1,1)
# while startTime <= dt.datetime.now():
#     startUnixTime = int(1000 * startTime.timestamp())
#     endUnixTime = int(1000 * (startTime + dt.timedelta(days=80)).timestamp())
#     withdraws.extend(binance_client.get_withdraw_history(startTime=startUnixTime, endTime=endUnixTime))
#     startTime += dt.timedelta(days=80)
# withdraws = pd.DataFrame(withdraws)


# folder = "C:/Users/PC/OneDrive/Documents/Finance/Tax/FY2022/"
# ada = pd.read_csv(f"{folder}ada_rewards.csv")
# ada["timestamp"] = pd.to_datetime(ada.date)
# for r in range(len(ada)):
#     ada.timestamp[r] = ada.timestamp[r].replace(tzinfo=None)
# ada.rename({"stake_rewards": "amount", "rate": "ADAAUD", "value": "AUD"}, axis=1, inplace=True)
# ada = ada[["timestamp", "amount", "ADAAUD", "AUD"]].copy()
# ada.to_parquet(f"{folder}ada_rewards.parquet")
# adaInt = ada.copy()

# ada = pd.read_csv(f"{folder}Cardano_transactions.csv")
# ada = ada[ada.TxHash.notna()].copy()
# ada.rename({"Koinly Date": "timestamp", "Fee Amount": "fee", "Amount": "amount"}, axis=1, inplace=True)
# ada = ada[["timestamp", "amount", "fee"]].copy()
# ada.timestamp = pd.to_datetime(ada.timestamp, dayfirst=True)
# for c in ["amount", "fee"]:
#     ada[c] = ada[c].str.replace(",", "")
#     ada[c] = ada[c].astype(float)/1000000
# ada = pd.concat([ada, adaInt], ignore_index=True)
# ada.sort_values("timestamp", inplace=True, ignore_index=True)
# ada.to_exc

# ada.sort_values("timestamp", inplace=True, ignore_index=True)
# for r in range(len(ada)):
#     ada.loc[r, "ADAAUD"] = get_aud_conversion("ADA", 1000*ada.timestamp[r].round(freq="T").timestamp())


# trades = pd.read_parquet(f"{folder}binance_trades_added_up.parquet")
# interest = pd.read_parquet(f"{folder}binance_interests.parquet")
# withdraws = pd.read_parquet(f"{folder}binance_withdraws.parquet")

# trades = trades[["AEST", "pair", "isBuyer", "price", "BUSDAUD", "USDTAUD", "BTCAUD", "BNBAUD", "qty", "baseQty", "commission"]].copy()
# trades["type"] = "trade"
# interest = interest[["AEST", "coin", "amount", "audRate"]].copy()
# interest["type"] = "interest"
# withdraws = withdraws[["AEST", "coin", "amount", "transactionFee"]].copy()
# withdraws["type"] = "withdrawal"
# withdraws["totalAmount"] = withdraws[["amount", "transactionFee"]].sum(axis=1)

# trans = pd.concat([interest, trades, withdraws], ignore_index=True)
# trans.sort_values("AEST", ignore_index=True, inplace=True)
# trans = trans[~((trans.type=="interest") & (trans.audRate.isna()))].reset_index(drop=True)
# trans.pair.replace("AGIBTC", "AGIXBTC", inplace=True)

# bal = pd.DataFrame({"AEST": dt.datetime(2021,3,19), "coin": "BUSD", "amount": 500/1.2978434, "audRate": 1.2978434, "comm": 0}, index=[0])
# pnl = pd.DataFrame()

# for r in range(len(trans)):
#     transType = trans.type[r]
#     if transType=="interest":
#         bal = pd.concat([bal, pd.DataFrame(trans.loc[r, ["AEST", "coin", "amount", "audRate"]]).transpose()], ignore_index=True)
#         bal.sort_values("AEST", ignore_index=True, inplace=True)
#     elif transType=="withdrawal":
#         coin = trans.coin[r]
#         withdrawn = trans.totalAmount[r]
#         coinTrans = bal[bal.coin==coin].sort_values("AEST")
#         coinTrans["cumAmount"] = coinTrans.amount.cumsum()
#         noTrim = any(coinTrans.cumAmount==withdrawn)
#         coinTrans = coinTrans[coinTrans.cumAmount>withdrawn].reset_index(drop=True)
#         coinTrans.sort_values("AEST", ignore_index=True, inplace=True)
#         if (not noTrim) & (len(coinTrans)>0):
#             coinTrans.loc[0, "amount"] = coinTrans.cumAmount[0] - withdrawn
#         bal = pd.concat([bal[bal.coin!=coin], coinTrans.drop("cumAmount", axis=1)], ignore_index=True)
#         bal.sort_values("AEST", ignore_index=True, inplace=True)
#     elif transType=="trade":
#         time = trans.AEST[r]
#         pair = trans.pair[r]
#         baseList = ["AUD", "BUSD", "BTC", "USDT"]
#         base = [b for b in baseList if pair.endswith(b)][0]
#         isBuy = trans.isBuyer[r]
#         price = trans.price[r] 
#         comm = trans.commission[r]
#         bnbrate = trans.BNBAUD[r]
        
#         if not((not isBuy) & (pair.endswith("AUD"))):
#             addBalCoin = pair.replace(base, "") if isBuy else base
#             addBalAmt = trans.qty[r] if isBuy else trans.baseQty[r]
#             addBalComm = 0 if ((not isBuy) & (base!="AUD")) else comm * bnbrate
#             if isBuy:
#                 addBalRate = price
#                 if base != "AUD":
#                     addBalRate *= trans[f"{base}AUD"][r]
#             else:
#                 addBalRate = 1/price if pair.startswith("AUD") else trans[f"{base}AUD"][r]
#             bal.loc[len(bal)] = [time, addBalCoin, addBalAmt, addBalRate, addBalComm]
        
#         if (isBuy & (base!="AUD")) or ((not isBuy) & (not pair.startswith("AUD"))):
#             rmBalCoin = base if isBuy else pair.replace(base, "")
#             rmBalQty = trans.baseQty[r] if isBuy else trans.qty[r]
#             coinTrans = bal[bal.coin==rmBalCoin].sort_values("AEST")
#             coinTrans["cumAmount"] = coinTrans.amount.cumsum()

#             noTrim = any(coinTrans.cumAmount==rmBalQty)
#             forBal = coinTrans[coinTrans.cumAmount>rmBalQty].copy()
#             forBal.sort_values("AEST", ignore_index=True, inplace=True)
#             if (not noTrim) & (len(forBal)>0):
#                 forBal.loc[0, "amount"] = forBal.cumAmount[0] - withdrawn
#             bal = pd.concat([bal[bal.coin!=rmBalCoin], forBal.drop("cumAmount", axis=1)], ignore_index=True)
#             bal.sort_values("AEST", ignore_index=True, inplace=True)

#             lastSellIdx = forBal.index.min()
#             if noTrim:
#                 lastSellIdx -= 1
#             forPnL = coinTrans.loc[0:lastSellIdx].copy()
#             forPnL.amount.clip(0, rmBalQty, inplace=True)
#             forPnL = forPnL[["coin", "amount", "AEST", "audRate", "comm"]].copy()
#             forPnL.rename({"AEST": "buyAEST", "audRate": "buyAUDRate", "comm": "buyComm"}, axis=1, inplace=True)
#             forPnL["sellAEST"] = time
#             if isBuy:
#                 sellRate = trans[f"{base}AUD"][r]
#             else:
#                 sellRate = price
#                 if base != "AUD":
#                     sellRate *= trans[f"{base}AUD"][r]
#             forPnL["sellAUDRate"] = sellRate
#             forPnL["sellComm"] = 0 if isBuy else comm * bnbrate * forPnL.amount / rmBalQty

#             pnl = pd.concat([pnl, forPnL], ignore_index=True)

# pnl["pnl"] = pnl.amount * (pnl.sellAUDRate - pnl.buyAUDRate) - pnl.buyComm.fillna(0) - pnl.sellComm
# pnl[pnl.sellAEST.between(dt.datetime(2021,7,1),dt.datetime(2022,7,1))].pnl.sum()
# pnl[pnl.sellAEST.between(dt.datetime(2021,7,1),dt.datetime(2022,7,1))].sort_values("pnl")

# trades = pd.DataFrame(trades)
# for c in ["price", "qty", "quoteQty", "commission"]:
#     trades[c] = trades[c].astype(float)
# trades.to_parquet("C:/Users/PC/OneDrive/Documents/Finance/Tax/FY2022/binance_trades.parquet")
# trades.to_parquet("C:/Users/PC/OneDrive/Documents/Finance/Tax/FY2022/binance_trades_added_up.parquet")

# trades["nearestUTC"] = trades.UTC.dt.round("1min")
# prices = trades.groupby(["nearestUTC", "symbol", "isBuyer"])["price", "BUSDAUD", "USDTAUD", "BTCAUD", "BNBAUD"].mean().reset_index()
# quants = trades.groupby(["nearestUTC", "symbol", "isBuyer"])["qty", "quoteQty", "commission"].sum().reset_index()
# trades = prices.merge(quants, on=["nearestUTC", "symbol", "isBuyer"], how="inner")

# trades["UTC"] = pd.to_datetime(trades.time, unit="ms")
# trades["AEST"] = trades.UTC + dt.timedelta(hours=10)
# trades.sort_values("AEST", ignore_index=True, inplace=True)

# for r in range(len(trades)):
#     trades.loc[r, "base_curr"] = trades.symbol[r][-3:]

# for r in range(len(trades)):
#     roundedTime = trades.UTC[r].round(freq="T")
#     roundedTime = 1000 * roundedTime.timestamp()
#     trades.loc[r, "BUSDAUD"] = get_aud_conversion("BUSD", roundedTime)
#     trades.loc[r, "USDTAUD"] = get_aud_conversion("USDT", roundedTime)
#     trades.loc[r, "BTCAUD"] = get_aud_conversion("BTC", roundedTime)
#     trades.loc[r, "BNBAUD"] = get_aud_conversion("BNB", roundedTime)
    

# start = dt.datetime(2021,3,1)
# end = start + dt.timedelta(days=28)
# startUnixTime = int(1000 * dt.datetime.timestamp(start))
# endUnixTime = int(1000 * dt.datetime.timestamp(end))
# binance_client.get_fiat_deposit_withdraw_history(transactionType="0-deposit", beginTime=startUnixTime, endTime=endUnixTime)



# histInt = pd.DataFrame(histInt)
# histInt["UTC"] = pd.to_datetime(histInt.divTime, unit="ms")
# histInt["nearestUTC"] = histInt.UTC.dt.round("1min")
# histInt["AEST"] = histInt.UTC + dt.timedelta(hours=10)
# histInt.sort_values("AEST", inplace=True)
# histInt = histInt[histInt.AEST.between(dt.datetime(2021,7,1),dt.datetime(2022,7,1))]
# histInt.drop_duplicates(inplace=True, ignore_index=True)

# histInt.to_parquet("C:/Users/PC/OneDrive/Documents/Finance/Tax/FY2022/binance_interests.parquet")

# histInt = pd.read_parquet("C:/Users/PC/OneDrive/Documents/Finance/Tax/FY2022/binance_interests.parquet")

# allTickers = pd.DataFrame(binance_client.get_all_tickers())





# for r in range(len(histInt)):
#     roundedTime = 1000 * histInt.nearestUTC[r].timestamp()
#     histInt.loc[r, "aud_rate"] = get_aud_conversion(histInt.asset[r], roundedTime)

# histInt["aud_interest"] = histInt.amount.astype(float) * histInt.aud_rate


# bnb_rewards = pd.read_csv("C:/users/PC/OneDrive/Documents/Finance/Cryptocurrency/bnb_staking_rewards.csv")

# bnb_rewards = pd.read_csv("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/bnb-staking-rewards_2023-09-24.csv")
# bnb_rewards["UTC"] = pd.to_datetime(bnb_rewards["Koinly Date"]).dt.tz_localize(None)
# bnb_rewards = bnb_rewards[bnb_rewards.UTC.between(dt.datetime(2022,6,30,14),dt.datetime(2023,6,30,14))].reset_index(drop=True)
# bnb_rewards.sort_values("UTC", inplace=True, ignore_index=True)

# for r in range(len(bnb_rewards)):
#     roundedTime = bnb_rewards.UTC[r].round(freq="T")
#     roundedTime = 1000 * roundedTime.timestamp()
#     bnb_rewards.loc[r, "aud_rate"] = binanceClient.get_aud_rate(bnb_rewards.Currency[r], roundedTime)

# bnb_rewards["aud_interest"] = bnb_rewards.Amount * bnb_rewards.aud_rate
# bnb_rewards.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2022/bnb_staking_rewards.parquet")

# bnb_rewards[bnb_rewards.UTC.between(dt.datetime(2021,6,30,14,0,0,0,pytz.UTC),dt.datetime(2022,6,30,14,0,0,0,pytz.UTC))].aud_interest.sum()


# ada = pd.read_csv("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2022/ada_rewards.csv")
# ada.date = pd.to_datetime(ada.date)
# for r in range(len(ada)):
#     ada.loc[r, "AEST"] = ada.date[r].astimezone("Australia/Brisbane")
#     ada.loc[r, "FY"] = ada.AEST[r].year if ada.AEST[r].month <= 6 else ada.AEST[r].year + 1

# ada.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2022/ada_rewards.parquet")

# ada.groupby("FY")["value"].sum()


# xtz = pd.read_excel("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2022/crypto.xlsx", sheet_name="tezos")
# xtz["UTC"] = xtz.AEST - dt.timedelta(hours=10)
# for r in range(len(xtz)):
#     roundedTime = xtz.UTC[r].round(freq="T")
#     roundedTime = 1000 * roundedTime.timestamp()
#     xtz.loc[r, "aud_rate"] = get_aud_conversion("XTZ", roundedTime)
# xtz["interest"] = xtz.payout * xtz.aud_rate
# xtz.interest.sum()
# xtz.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2022/xtz.parquet")

# histInt = pd.DataFrame(binance_client.get_asset_dividend_history(startTime=startTime, endTime=endTime, limit=500)["rows"])
# histInt.divTime = pd.to_datetime(histInt.divTime, unit="ms") + dt.timedelta(hours=10)

# interest = pd.DataFrame(binance_client.get_asset_dividend_history()["rows"])
# interest["UTC"] = pd.to_datetime(interest.divTime, unit="ms")

