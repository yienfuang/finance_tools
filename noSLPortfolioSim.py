"""
This script takes the simulated trades from a parquet file, groups them by specific parameters, and simulates a trading strategy without stop-losses.
It processes each group of trades in parallel, calculating entry and exit sizes based on a portfolio size tracker.
It logs the simulation process and saves the results to parquet files for each group of parameters.
"""


import numpy as np
import pandas as pd

# plotly
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