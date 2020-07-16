import yfinance as yf

df = yf.Ticker('TSLA').history(period='1wk', interval='1m')
print(df)