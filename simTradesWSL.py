import datetime as dt
import logging
import numpy as np
import pandas as pd

# multiprocessing
from itertools import product
from multiprocessing import Pool
from contextlib import closing

rawPrices = pd.read_parquet("D:/binanceSMAPrices.parquet")

buySMACombo = [3, 7, 14, 30, 200]
abovePCForBuyCombo = [0.05, 0.1, 0.15, 0.2]
sellSMACombo = [3, 7, 14, 30, 200]
belowPCForSellCombo = [0.05, 0.1, 0.15, 0.2]
slCombo = [0.05, 0.1, 0.15, 0.2]
startDateCombo = pd.date_range(rawPrices.openTime.min(), rawPrices.openTime.max() - dt.timedelta(days=365), freq=dt.timedelta(days=90))
parameterCombos = product(buySMACombo, abovePCForBuyCombo, sellSMACombo, belowPCForSellCombo, slCombo, startDateCombo)

positionsList = []
portfolioSizeList = []

logger = logging.getLogger()
handler = logging.FileHandler("D:/simTradesWSL.log")
logger.addHandler(handler)

def simTradesWSL(buySMA, abovePCForBuy, sellSMA, belowPCForSell, sl, startDate):
    
    logger.error(f"Simulating buySMA:{buySMA} abovePCForBuy:{abovePCForBuy} sellSMA:{sellSMA} belowPCForSell:{belowPCForSell} sl:{sl}")

    # change start date of raw prices
    price = rawPrices[rawPrices.openTime >= startDate].reset_index(drop=True)

    portfolioSize = pd.DataFrame({"date": price.openTime.min(), "portfolioSize": 1}, index=[0])
    positions = pd.DataFrame(columns=["coin", 'buyTime', 'buyPrice', 'sellPrice', 'sellTime', 'profit', 'entrySize', 'exitSize'])
    buySMAColName = f"is{int(abovePCForBuy*100)}pcAboveSMA{buySMA}"
    sellSMAColName = f"is{int(belowPCForSell*100)}pcBelowSMA{sellSMA}"
    tradeTime = price[price[buySMAColName]==1].openTime.min()
    maxOpenTrades = 10

    while tradeTime <= price.openTime.max():
        
        # number of open trades
        noOpenTrades = 0 if len(positions)==0 else positions.exitSize.isna().sum()

        if noOpenTrades > 0:

            slCheck = positions[positions.exitSize.isna()].copy()
            priceSubset = price.loc[price.openTime==tradeTime, ["coin", "openTime", "low", "close", f"{sellSMAColName}"]]
            slCheck = slCheck.merge(priceSubset, how="left", on="coin")
            # if lowest price of the day is below stop-loss threshold, sell
            # otherwise, if the sell signal is triggered, sell
            # if both conditions are not met, do nothing
            slCheck["sellPrice"] = np.where(slCheck.low <= (1-sl)*slCheck.buyPrice, (1-sl)*slCheck.buyPrice,
                                            np.where(slCheck[f"{sellSMAColName}"], slCheck.close, np.nan))
            slCheck = slCheck[slCheck.sellPrice.notna()].reset_index(drop=True)
            if len(slCheck) > 0:
                slCheck.drop(columns="sellTime", inplace=True)
                slCheck.rename({"openTime": "sellTime"}, axis=1, inplace=True)
                slCheck["profit"] = slCheck.sellPrice/slCheck.buyPrice - 1
                slCheck["exitSize"] = slCheck.entrySize*(1+slCheck.profit)
                slCheck.reset_index(drop=True, inplace=True)
                # ID made by concatenating coin and entry time to ensure uniqueness
                # slCheckID = list(slCheck.coin + slCheck.buyTime.astype(str))

                for pr in range(len(slCheck)):
                    coin = slCheck.iloc[pr].coin
                    buyTime = slCheck.iloc[pr].buyTime
                    for oc in ["sellPrice", "sellTime", "profit", "exitSize"]:
                        positions.loc[(positions.coin==coin) & (positions.buyTime==buyTime), oc] = slCheck.iloc[pr][oc]

                # positions = pd.concat([positions[~(positions.coin + positions.buyTime.astype(str)).isin(slCheckID)], slCheck[positions.columns]], ignore_index=True)

                portfolioSize.loc[len(portfolioSize), ["date", "portfolioSize"]] = [
                    tradeTime,
                    portfolioSize.portfolioSize.iloc[-1] * (1 - len(slCheck) / maxOpenTrades) # adjusted size of open trades
                    + slCheck.exitSize.sum() # realised size of closed trades
                    ]

                noOpenTrades -= len(slCheck)

        # possible entries
        possibleBuys = price[(price.openTime == tradeTime) & (price[buySMAColName]==1)].reset_index(drop=True)
        if len(possibleBuys) > 0:
            
            # number of possible buys
            noPossibleBuys = max(0, maxOpenTrades - noOpenTrades)

            if noPossibleBuys > 0:
                possibleBuys = possibleBuys.nlargest(noPossibleBuys, "volume").reset_index(drop=True)

                entrySize = .1 * portfolioSize.portfolioSize.iloc[-1]
                possibleBuys["entrySize"] = entrySize
                possibleBuys.rename({"openTime": "buyTime", "close": "buyPrice"}, axis=1, inplace=True)

                for pr in range(len(possibleBuys)):
                    positions.loc[len(positions), ["coin", "buyTime", "buyPrice", "entrySize"]] = possibleBuys.iloc[pr][["coin", "buyTime", "buyPrice", "entrySize"]]

                # print(f"Added {len(possibleBuys)} trades at {tradeTime} with entry size {entrySize}")

        tradeTime += dt.timedelta(days=1)

    positions[["buySMA", "abovePCForBuy", "sellSMA", "belowPCForSell", "sl", "startDate"]] = [buySMA, abovePCForBuy, sellSMA, belowPCForSell, sl, startDate]
    portfolioSize[["buySMA", "abovePCForBuy", "sellSMA", "belowPCForSell", "sl", "startDate"]] = [buySMA, abovePCForBuy, sellSMA, belowPCForSell, sl, startDate]
    positions.to_parquet(f"D:/temp/positions_{buySMA}_{int(abovePCForBuy*100)}pcAboveSMA_{sellSMA}_{int(belowPCForSell*100)}pcBelowSMA_{sl}_startDate{startDate.strftime('%Y%m%d')}.parquet")
    portfolioSize.to_parquet(f"D:/temp/portfolioSize_{buySMA}_{int(abovePCForBuy*100)}pcAboveSMA_{sellSMA}_{int(belowPCForSell*100)}pcBelowSMA_{sl}_startDate{startDate.strftime('%Y%m%d')}.parquet")

    logger.error(f"Simulation completed for buySMA:{buySMA} abovePCForBuy:{abovePCForBuy} sellSMA:{sellSMA} belowPCForSell:{belowPCForSell} sl:{sl} startDate:{startDate}. Trades and portfolio sized appended")

if __name__ == "__main__":

    with closing(Pool(12)) as p:
        p.starmap(simTradesWSL, parameterCombos)