import yfinance as yf
import pandas as pd
import numpy as np


class DataPreparation:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self):
        self.data = yf.download(
            self.ticker, 
            start=self.start_date, 
            end=self.end_date,
            multi_level_index = False
            )
    def get_returns(self):
        # self.data['LogReturns'] = np.log(self.data['Close'] / self.data['Open'])
        # self.data['Returns'] = self.data['Close'] / self.data['Open']
        # self.data.drop(columns=['Adj Close'], inplace=True)

    def get_final_dataset(self):
        self.get_data()
        self.get_returns()
        return self.data