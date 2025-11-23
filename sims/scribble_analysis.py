import datetime as dt
import logging
import numpy as np
import os
import pandas as pd


folderPath = "D:/temp/"
files = [f for f in os.listdir(folderPath) if f.startswith("portfolioSize_") and f.endswith(".parquet")]
allDF = []
for file in files:
    filePath = os.path.join(folderPath, file)
    df = pd.read_parquet(filePath)
    allDF.append(df)
allDF = pd.concat(allDF, ignore_index=True)
allDF["id"] = allDF.buySMA.astype(str) + "_" + allDF.abovePCForBuy.astype(str) + "_" + allDF.sellSMA.astype(str) + "_" + allDF.belowPCForSell.astype(str) + "_" + allDF.sl.astype(str) + "_" + allDF.startDate.astype(str)

lastDates = allDF.groupby("id")["date"].max().reset_index()
allDF = allDF.merge(lastDates, on=["id", "date"], how="inner")

allDF["stratID"] = allDF.buySMA.astype(str) + "_" + allDF.abovePCForBuy.astype(str) + "_" + allDF.sellSMA.astype(str) + "_" + allDF.belowPCForSell.astype(str) + "_" + allDF.sl.astype(str)
stats = allDF.groupby("stratID")["portfolioSize"].describe().reset_index()

allDF = pd.read_parquet("D:/portfolioSizeSim_SMAsWSL_diffStarts.parquet")
allDF = allDF[allDF.date<=allDF.startDate+dt.timedelta(days=365)].reset_index(drop=True)
lastDates = allDF.groupby("id")["date"].max().reset_index()
allDF = allDF.merge(lastDates, on=["id", "date"], how="inner")
bestPerf = allDF.groupby("startDate")["portfolioSize"].max().reset_index()
bestPerf = bestPerf.merge(allDF, on=["startDate", "portfolioSize"], how="inner")


folderPath = "D:/temp_allTimeStart/"
files = [f for f in os.listdir(folderPath) if f.startswith("portfolioSize_") and f.endswith(".parquet")]

allDF = []
for file in files:
    filePath = os.path.join(folderPath, file)
    df = pd.read_parquet(filePath)
    allDF.append(df)

allDF = pd.concat(allDF, ignore_index=True)
allDF["id"] = allDF.buySMA.astype(str) + "_" + allDF.abovePCForBuy.astype(str) + "_" + allDF.sellSMA.astype(str) + "_" + allDF.belowPCForSell.astype(str) + "_" + allDF.sl.astype(str)

portfolioMax = allDF.groupby("id")["portfolioSize"].max().reset_index()
portfolioMax = portfolioMax.merge(allDF[list(portfolioMax.columns) + ["date"]], on=["id", "portfolioSize"], how="inner")

allDF = allDF.merge(portfolioMax, on=["id"], how="inner")
allDF = allDF[allDF.date_x > allDF.date_y].reset_index(drop=True)

portfolioMin = allDF.groupby("id")["portfolioSize_x"].min().reset_index()
allDF = allDF.merge(portfolioMin, on=["id", "portfolioSize_x"], how="inner")
allDF["allTimeDrawdown"] = allDF.portfolioSize_x / allDF.portfolioSize_y - 1

import plotly.express as px
fig = px.line(allDF, x="date", y="portfolioSize", color="id", title="Portfolio Size Over Time")

lastDate = allDF.groupby("id")["date"].max().reset_index()
allDF = allDF.merge(lastDate, on=["id", "date"], how="inner")

portfolioHigh = allDF.groupby("id")["portfolioSize"].max().reset_index()
allDF = allDF.merge(portfolioHigh, on=["id", "portfolioSize"], how="inner")

allDF["year"] = allDF.date.dt.year


files = [f for f in os.listdir("D:/simTradesNPortSize_noSL") if f.startswith("portfolioSize") and f.endswith(".parquet")]
allDF = []
for file in files:
    filePath = os.path.join("D:/simTradesNPortSize_noSL", file)
    df = pd.read_parquet(filePath)
    allDF.append(df)

allDF = pd.concat(allDF, ignore_index=True)
allDF["id"] = allDF.MAForBuy.astype(str) + "_" + allDF.abovePCForBuy.astype(str) + allDF.MAForSell.astype(str) + allDF.belowPCForSell.astype(str)
