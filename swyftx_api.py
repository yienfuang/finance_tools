import datetime as dt
from dateutil.relativedelta import relativedelta as relDelta
import pandas as pd
import requests
import time

def getAUDPrice(coin:str="BTC", startTime=dt.datetime.now(), endTime=None, timezone:str="UTC", resolution="1m", ohlc:str="close", bidAsk:str="bid"):
    """
    Get 1-minute price

    Parameters
    ----------
    coin     : Coin ticker, e.g. BTC, ETH, BNB
    time     : Can be datetime/pandas.Timestamp object or str in yyyy-mm-dd hh:mm:ss format
    timezone : Timezone in str e.g. 'UTC', 'Australia/Melbourne', 'Australia/Brisbane'
    ohlc     : "open", "high", "low" or "close". Anything else, the whole json body is returned
    bidAsk   : "bid" or "ask". Usually it will be bid, especially for staking rewards
    """
    assert bidAsk in ["bid", "ask"], "bidAsk should be 'bid' or 'ask'"

    validRes = ["1m", "5m", "1h", "4h", "1d"]
    assert resolution in validRes, f"resolution needs to be one of {', '.join(validRes)}"
    roundMapping = {"1m": "T", "5m": "5T", "1h": "H", "4h": "4H", "1d": "D"}

    # round timestamp down to nearest minute
    timeStart = pd.Timestamp(startTime, tz=timezone).round(freq=roundMapping[resolution])

    # API takes UNIX timestamp in milliseconds
    timeStart = int(1000*timeStart.timestamp())

    # allocate some block to the query
    timeEnd   = pd.Timestamp(endTime, tz=timezone).round(freq=roundMapping[resolution]) if endTime else timeStart + 4*60*60*1000
    if endTime:
        timeEnd = int(1000*timeEnd.timestamp())

    # make API call and return price
    apiURL = f"https://api.swyftx.com.au/charts/getBars/AUD/{coin}/{bidAsk}/?resolution={resolution}&timeStart={timeStart}&timeEnd={timeEnd}&limit=20000"
    price = requests.get(apiURL).json()

    # return relevant price
    return price["candles"][0][ohlc] if ohlc in ["open", "high", "low", "close"] else price

startTime = dt.datetime(2016,12,26)
prices = []
startOfLoop = dt.datetime.now()
while startTime < startOfLoop:
    endTime = min(dt.datetime.now(), startTime + relDelta(years=2))
    prices.extend(getAUDPrice(coin="WOO", startTime=startTime, endTime=endTime, resolution="1h", ohlc="", bidAsk="ask")["candles"])
    startTime = endTime
prices = pd.DataFrame(prices)
prices["UTC"] = pd.to_datetime(prices.time, unit="ms")
prices["hourOfDay"] = prices.UTC.dt.hour
prices["dayOfWeek"] = prices.UTC.dt.dayofweek
prices["id"] = prices.dayOfWeek.astype(str) + "_" + prices.hourOfDay.astype(str)
dailies = prices.groupby("hourOfDay")["open"].mean().reset_index()
dailies.sort_values("open", inplace=True)
avgID = prices.groupby("id")["open"].mean().reset_index()
avgID.sort_values("open", inplace=True)
avgID

avgPrice = prices.groupby("hourOfDay")["open"].mean().reset_index()
avgID = prices[prices.dayOfMonth<=28].groupby("id")["open"].mean().reset_index()
avgID.sort_values("open", inplace=True)

prices["dateBin"] = prices.UTC.dt.date
minOfDay = prices.groupby("dateBin")["open"].min().reset_index()
hours = minOfDay.merge(prices, how="inner", on=["dateBin", "open"])


# Binance staking rewards
df = pd.read_csv("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/bnb-staking-rewards_2023-09-24.csv")
df["UTC"] = pd.to_datetime(df["Koinly Date"]).dt.tz_localize(None)
df = df[df.UTC.between(dt.datetime(2022,6,30,14),dt.datetime(2023,6,30,14))]
df = df.groupby("UTC")["Amount"].sum().reset_index()
for r in range(200, len(df)):
    df.loc[r, "BNBPrice"] = get1mAUDPrice(coin="BNB", time=df.loc[r, "UTC"])
df["AUD"] = df.Amount * df.BNBPrice
df.AUD.sum()
df.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/bnb_staking_rewards.parquet")

# Mina staking rewards
df = pd.read_csv("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/minaexplorer-tx.csv")
df["UTC"] = pd.to_datetime(df.Date)
df = df[df.From.str.endswith("thYLsz")].reset_index(drop=True)
df = df[["UTC", "Amount"]].copy()
for r in range(len(df)):
    df.loc[r, "MINAPrice"] = get1mAUDPrice(coin="MINA", time=df.loc[r, "UTC"])
df["AUD"] = df.Amount * df.MINAPrice
df[df.UTC<=dt.datetime(2023,6,30,14)].AUD.sum()
df.to_parquet("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/mina_staking_rewards.parquet")

# XTZ staking rewards
df = pd.read_csv("C:/users/PC/OneDrive/Documents/Finance/Tax/FY2023/tezos_statement_fy23.csv")
df = df[df["From address"].notna()].reset_index(drop=True)
df = df[df["From address"].str.endswith("GYKi")].reset_index(drop=True)
df["UTC"] = pd.to_datetime(df.Datetime)
df = df[["UTC", "Received"]].copy()
for r in range(len(df)):
    try:
        df.loc[r, "XTZPrice"] = get1mAUDPrice(coin="XTZ", time=df.loc[r, "UTC"])
    except:
        print(f"Didn't work for row {r} at time {df.loc[r, 'UTC']}")
df.loc[df.XTZPrice.isna(), "XTZPrice"] = get1mAUDPrice(coin="XTZ", time=f"2023-04-04 22:00")
df["AUD"] = df.Received * df.XTZPrice
df.AUD.sum()

