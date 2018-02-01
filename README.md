### Predict stock market prices using RNN

Check my blog post "Predict Stock Prices Using RNN": [Part 1](https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html) and [Part 2](https://lilianweng.github.io/lil-log/2017/07/22/predict-stock-prices-using-RNN-part-2.html) for the tutorial associated.

One thing I would like to emphasize that because my motivation is more on demonstrating how to build and train an RNN model in Tensorflow and less on solve the stock prediction problem, I didn't try too hard on improving the prediction outcomes. You are more than welcome to take this repo as a reference point and add more stock prediction related ideas to improve it. Enjoy.

1. Make sure `tensorflow` has been installed.
2. First download the full S&P 500 data from [Yahoo! Finance ^GSPC](https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC) (click the "Historical Data" tab and select the max time period). And save the .csv file to `data/SP500.csv`.
3. Run `python data_fetcher.py` to download the prices of individual stocks in S & P 500, each saved to `data/{{stock_abbreviation}}.csv`.
(NOTE: Google Finance API returns the prices for 4000 days maximum. If you are curious about the data in even early times, try modify `data_fetcher.py` code to send multiple queries for one stock. Here is the data archive ([stock-data-lilianweng.tar.gz](https://drive.google.com/open?id=1QKVkiwgCNJsdQMEsfoi6KpqoPgc4O6DD)) of stock prices I crawled up to Jul, 2017. Please untar this file to replace the "data" folder in the repo for test runs.)
4. Run `python main.py --help` to check the available command line args.
5. Run `python main.py` to train the model.


For examples,
- Train a model only on SP500.csv; no embedding
```bash
python main.py --stock_symbol=SP500 --train --input_size=1 --lstm_size=128 --max_epoch=50
```

- Train a model on 100 stocks; with embedding of size 8
```bash
python main.py --stock_count=100 --train --input_size=1 --lstm_size=128 --max_epoch=50 --embed_size=8
```

- Start your Tensorboard
```bash
cd stock-rnn
mkdir logs
tensorboard --logdir ./logs --port 1234 --debug
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
