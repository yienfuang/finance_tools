import datetime as dt
from dateutil import relativedelta
import math
import numpy as np
import pandas as pd
import plotly.express as px

class Share:

    def __init__(self, brokerage):
        self.brokerage = brokerage
    
    def get_entry_price(self, capital, fairPrice):
        """
        Calculate entry price for a stock, to cover brokerage
        """

        # calculate intended number of shares to buy
        nShares = math.floor(capital/fairPrice)
        
        # find entry price to cover brokerage
        entryPrice = ((capital - self.brokerage) / nShares) // .01 / 100
        entryPrice = min(entryPrice, fairPrice)

        # include extra shares that can be bought with cheaper strike price
        cost = nShares * entryPrice + self.brokerage
        nShares += math.floor((capital - cost)/entryPrice)

        # print result
        print("Buy {} shares at {}. Total cost: {}".format(nShares, entryPrice, nShares * entryPrice + self.brokerage))
    
    def get_stop_loss_cover_scenarios(self, nShares, entryPrice, stopLoss, coverEntryBrokerage=True):
        
        df = pd.DataFrame({"n_shares_to_sell": [s for s in range(1, nShares)]})
        df["remaining_shares"] = nShares - df.n_shares_to_sell
        df["stop_loss_amount"] = stopLoss * entryPrice * df.remaining_shares
        df.stop_loss_amount = df.stop_loss_amount.round(2)

        nTrades = 3 if coverEntryBrokerage else 2
        totalBrokerage = nTrades * self.brokerage

        df["stop_loss_amount_w_brokerage"] = df.stop_loss_amount + totalBrokerage
        df["exit_price"] = entryPrice + df.stop_loss_amount_w_brokerage / df.n_shares_to_sell

        return df


def CalculateMortgageEnd(mortgageRemainder, mortgageStart, repaymentAmount,
    interestRate, mortgageTS=dt.date.today(), returnDF=True):
    repaymentDay = mortgageStart.day
    if returnDF:
        df = pd.DataFrame({"date": mortgageTS, "remainder": mortgageRemainder}, index=[0])
    while mortgageRemainder > 0:
        if mortgageTS.day < repaymentDay:
            mortgageTS = mortgageTS.replace(day=repaymentDay)
            mortgageRemainder -= repaymentAmount
        else:
            mortgageTS += relativedelta.relativedelta(months=1)
            mortgageTS = mortgageTS.replace(day=1)
            mortgageTS -= dt.timedelta(days=1)
            eom = mortgageTS.day
            fy = mortgageTS.year if mortgageTS.month <= 6 else mortgageTS.year + 1
            numDays = 366 if fy % 4 == 0 else 365
            mortgageRemainder += round((repaymentDay * (mortgageRemainder + repaymentAmount) + (eom - repaymentDay) * mortgageRemainder) / numDays * interestRate, 2)
            mortgageTS += dt.timedelta(days=1)

        if returnDF:
            df = pd.concat([df, pd.DataFrame({"date": mortgageTS, "remainder": mortgageRemainder}, index=[0])], ignore_index=True)

    if returnDF:
        return df
    else:
        print("Mortgage finishes on {}".format(mortgageTS.strftime("%Y-%m-%d")))