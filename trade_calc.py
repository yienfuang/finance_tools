import datetime as dt
import numpy as np
import pandas as pd

cs = pd.read_csv("C:/users/PC/downloads/orderhistory.csv")
cs.columns = ["datetime", "type", "pair", "coinAmount", "rateWFee", "rateWOFee", "fee", "feeAUD", "gstAUD", "totalAUD", "totalAUDWGST"]
cs.datetime = pd.to_datetime(cs.datetime, dayfirst=True)
cs["fy"] = cs.datetime.dt.to_period("Q-JUN").dt.qyear

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
        subset = tradeDF[(tradeDF.soldOrSent.isna()) & (tradeDF.coin==coin)]
        if len(subset)==0:
            newRowNum = len(df)
            tradeDF.loc[newRowNum, ["coin", "coinAmount", "soldOrSent", "sellDateTime"]] = [coin, amt, soldOrSent, df.datetime[r]]
            if txType == "Sell":
                tradeDF.loc[newRowNum, ["sellPrice", "sellFee"]] = [df.rateWOFee[r], df.feeAUD[r]]
        else:
            rowNum = subset.index[0]
            entryAmt = tradeDF.loc[rowNum, "coinAmount"]
            if entryAmt >= amt:
                if entryAmt > amt:
                    newRowNum = len(df)
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
                    tradeDF.loc[rowNum, ["sellPrice", "sellFee"]] = [df.rateWOFee[r], df.feeAUD[r]]
            else: