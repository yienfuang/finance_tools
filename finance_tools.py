import datetime as dt
from dateutil import relativedelta
import math
import numpy as np
import pandas as pd
import plotly.express as px

class Share:

    def __init__(self, brokerage):

        # brokerage fee per trade ($)
        self.brokerage = brokerage
    
    def get_entry_price(self, capital, fairPrice):
        """
        Calculate entry price for a stock, to cover for brokerage

        Parameters
        ----------
        capital (float)   : amount of money intended for investment
        fairPrice (float) : fair price to buy share at (excl. brokerage)
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
        """
        Get scenarios of trimming down positions to cover for stop loss

        Parameters
        ----------
        nShares (int)              : number of shares held
        entryPrice (float)         : price shares were bought at
        stopLoss (frac)            : stop loss arranged, in fraction
        coverEntryBrokerage (bool) : if True, cover for brokerage at entry, too; else, only cover for brokerages for profit taking and stop loss
        """

        # all possible number of shares to sell
        df = pd.DataFrame({"n_shares_to_sell": [s for s in range(1, nShares)]})

        # number of shares remaining
        df["remaining_shares"] = nShares - df.n_shares_to_sell

        # projected loss at stop
        df["stop_loss_amount"] = stopLoss * entryPrice * df.remaining_shares
        df.stop_loss_amount = df.stop_loss_amount.round(2)

        # total amount of brokerage to cover
        nTrades = 3 if coverEntryBrokerage else 2
        totalBrokerage = nTrades * self.brokerage

        # projected loss plus brokerage at stop
        df["stop_loss_amount_w_brokerage"] = df.stop_loss_amount + totalBrokerage

        # get profit taking price
        df["exit_price"] = entryPrice + df.stop_loss_amount_w_brokerage / df.n_shares_to_sell
        df.exit_price = df.exit_price.round(3)

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


def pickShares(companyList="C:/users/a30004759/downloads/asx202312.csv"):
    """
    1. Get company list from https://www.asx.com.au/markets/trade-our-cash-market/directory
    2. Save it somewhere locally and point the function to it
    """

    df = pd.read_csv(companyList)

    # convert market cap into float
    df.rename({"Market Cap": "market_cap"}, axis=1, inplace=True)
    df = df[df.market_cap.notna()].reset_index(drop=True)
    df = df[df.market_cap.str[0].isin(np.arange(1,10).astype(str))].reset_index(drop=True)
    df.market_cap = df.market_cap.astype(float)

    # pick the ASX 500 companies
    df.sort_values("market_cap", ascending=False, inplace=True, ignore_index=True)
    df = df.loc[:499].copy()

    return df[df.index.isin(np.random.choice(np.arange(1,501), size=4, replace=False))]
