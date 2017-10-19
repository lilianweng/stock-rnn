### Predict stock market prices using RNN

Check my blog post **["Predict Stock Prices Using RNN: Part 1"](https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN.html)** for the tutorial associated.

1. Make sure `tensorflow` has been installed.
2. First download the full S&P 500 data from [Yahoo! Finance ^GSPC](https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC) (click the "Historical Data" tab and select the max time period). And save the .csv file to `data/SP500.csv`.
3. Run `python data_fetcher.py` to download the prices of individual stocks in S & P 500, each saved to `data/{{stock_abbreviation}}.csv`.
4. Run `python main.py --help` to check the available command line args.
5. Run `python main.py` to train the model.


For examples,
- Train a model only on SP500.csv; no embedding
```
python main.py --stock_symbol=SP500 --train --input_size=5 --lstm_size=128 --max_epoch=50
```

- Train a model on 100 stocks; with embedding of size 8
```
python main.py --stock_count=100 --train --input_size=5 --lstm_size=128 --max_epoch=50 --embed_size=8
```

My python environment:
```
BeautifulSoup==3.2.1
numpy==1.13.1
pandas==0.16.2
scikit-learn==0.16.1
scipy==0.19.1
tensorflow==1.2.1
urllib3==1.8
```
