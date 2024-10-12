import datetime as dt
import numpy as np
import pandas as pd

# Current code only for coinspot trades

# get all order history from coinspot website or API
cs = pd.read_csv("C:/users/PC/downloads/orderhistory.csv")
cs.columns = ["datetime", "type", "pair", "coinAmount", "rateWFee", "rateWOFee", "fee", "feeAUD", "gstAUD", "totalAUD", "totalAUDWGST"]
cs.datetime = pd.to_datetime(cs.datetime, dayfirst=True)

# get all sends and receives from coinspot website or API
sr = pd.read_csv("C:/users/PC/downloads/sendsreceives.csv")
sr.columns = ["datetime", "type", "coin", "status", "fee", "amount", "address", "txid", "aud"]
sr.datetime = pd.to_datetime(sr.datetime, dayfirst=True)
sr.amount = sr.amount.abs()

df = pd.concat([cs, sr], ignore_index=True)
df.sort_values("datetime", inplace=True, ignore_index=True)

tradeDF = pd.DataFrame(columns=[
    "coin", "coinAmount",
    "boughtOrReceived", "soldOrSent",
    "buyDateTime", "buyPrice", "buyFee",
    "sellDateTime", "sellPrice", "sellFee"])
for r in range(len(df)):
    txType = df.type[r]
    coin = df.pair[r].replace("/AUD", "") if txType in ["Buy", "Sell"] else df.coin[r]
    amt = df.coinAmount[r] if txType in ["Buy", "Sell"] else df.amount[r]
    if txType in ["Buy", "Receive"]:
        boughtOrReceived = "bought" if txType == "Buy" else "received"
        rowNum = len(tradeDF)
        tradeDF.loc[rowNum, ["coin", "coinAmount", "boughtOrReceived", "buyDateTime"]] = [coin, amt, boughtOrReceived, df.datetime[r]]
        if boughtOrReceived == "bought":
            tradeDF.loc[rowNum, ["buyPrice", "buyFee"]] = [df.rateWOFee[r], df.feeAUD[r]]
    elif txType in ["Sell", "Send"]:
        soldOrSent = "sold" if txType == "Sell" else "sent"
        subset = tradeDF[(tradeDF.soldOrSent.isna()) & (tradeDF.coin==coin)].copy()
        subset["cumEntryAmt"] = subset.coinAmount.cumsum()
        if amt in subset.cumEntryAmt:
            subset = subset[subset.cumEntryAmt<=amt].copy()
        else:
            subset = subset[subset.cumEntryAmt.shift(fill_value=0)<=amt].copy()
        for s in range(len(subset)-1):
            tradeDF.loc[subset.index[s], ["soldOrSent", "sellDateTime"]] = [soldOrSent, df.datetime[r]]
            if soldOrSent == "sold":
                tradeDF.loc[subset.index[s], ["sellPrice", "sellFee"]] = [df.rateWOFee[r], df.feeAUD[r] * tradeDF.loc[subset.index[s], "coinAmount"] / amt]
        if len(subset)==0:
            newRowNum = len(tradeDF)
            tradeDF.loc[newRowNum, ["coin", "coinAmount", "soldOrSent", "sellDateTime"]] = [coin, amt, soldOrSent, df.datetime[r]]
            if soldOrSent == "sold":
                tradeDF.loc[newRowNum, ["sellPrice", "sellFee"]] = [df.rateWOFee[r], df.feeAUD[r]]
        else:
            rowNum = subset.index.max()
            entryAmt = tradeDF.loc[rowNum, "coinAmount"]
            amt = amt if len(subset)==1 else amt - subset[:-1].cumEntryAmt.max()
            if entryAmt > amt:
                newRowNum = len(tradeDF)
                tradeDF.loc[newRowNum] = tradeDF.loc[rowNum]
            tradeDF.loc[rowNum, ["soldOrSent", "sellDateTime"]] = [soldOrSent, df.datetime[r]]
            if entryAmt > amt:
                tradeDF.loc[rowNum, "coinAmount"] = amt
                tradeDF.loc[newRowNum, "coinAmount"] = entryAmt - amt
                if soldOrSent == "sold":
                    buyFee = tradeDF.loc[rowNum, "buyFee"]
                    tradeDF.loc[rowNum, "buyFee"] = buyFee * amt / entryAmt
                    tradeDF.loc[newRowNum, "buyFee"] = buyFee * (1 - amt / entryAmt)
                tradeDF.sort_values(["buyDateTime", "sellDateTime"], ignore_index=True, inplace=True)
            if soldOrSent == "sold":
                tradeDF.loc[rowNum, ["sellPrice", "sellFee"]] = [df.rateWOFee[r], df.feeAUD[r] * tradeDF.loc[rowNum, "coinAmount"] / amt]

tradeDF.sellDateTime = pd.to_datetime(tradeDF.sellDateTime)
tradeDF["fy"] = tradeDF.sellDateTime.dt.to_period("Q-JUN").dt.qyear
tradeDF["pnl"] = tradeDF.coinAmount * (tradeDF.sellPrice - tradeDF.buyPrice) - tradeDF.buyFee - tradeDF.sellFee
tradeDF.groupby("fy")["pnl"].sum()