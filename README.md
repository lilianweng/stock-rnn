### Predict stock market prices using RNN

Check my blog post **["Predict Stock Prices Using RNN: Part 1"](https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN.html)** for the tutorial associated.

1. Make sure `tensorflow` has been installed.
2. First download the full S&P 500 data from [Yahoo! Finance ^GSPC](https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC) or [this link](https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=-630950400&period2=1499756400&interval=1d&events=history&crumb=E5vf6Xp7xTo). And save the .csv file to `_data/SP500.csv`.
3. Run `python main.py --help` to check the available command line args.
4. Run `python main.py` to train the model.

Such as:
```
python main.py --stock_symbol=SP500 --train --input_size=5 --lstm_size=128 --max_epoch=50
```
