from data_preparation import DataPreparation

def main(tickers, start_date, end_date):
    data_preparation = DataPreparation(tickers, start_date, end_date)
    data = data_preparation.get_final_dataset()
    print(data)

if __name__ == "__main__":
    tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    main(tickers, start_date, end_date)