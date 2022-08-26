# TAGNN

## Paper & Dataset
This is the code for the SIGIR 2020 Paper: [TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation](https://arxiv.org/abs/2005.02844).

I implemented with Dressip which is the dataset for [RecSys Challenge 2022](http://www.recsyschallenge.com/2022/). Download the dressipi dataset and put it in `datasets/`.  
The original used datasets in TAGNN are YOOCHOOSE (RecSys Challenge 2015) and DIGINETICA.

**Notes:**  
Actually, the dataset I used is `train_sessions_purchases.csv` which is not in the official dressipi dataset.  
This csv is the combination of `train_sessions.csv` and `train_purchases.csv`. I put each session in `train_purchases.csv` at the end of `train_session.csv` according to their `session_id` to generate `train_sessions_purchases.csv` (Thanks to my group member).  

For example:  
This  
```
session_id,item_id,date
3,9655,2020-12-18 21:25:00.373
3,9655,2020-12-18 21:19:48.093
```
becomes
```
session_id,item_id,date
3,9655,2020-12-18 21:25:00.373
3,9655,2020-12-18 21:19:48.093
3,15085,2020-12-18 21:26:47.986 <- this is the purchase of this session
```

There is a small dataset sample included in the folder `datasets/`, which can be used to test the correctness of the code.


## Requirments
- Python 3
- PyTorch 1.4.0


## Usage
### Dataset
You need to run the datasets/preprocess.py first to generate the correct data format.
```
python3 preprocess.py --dataset dressipi --date 2021-04 --cut

optional arguments:
--dataset: dataset name
--date: start date of training dataset
--cut: cut the test dataset
```
**Tips:**  
The generated datasets are binary file, you can't read them.  
Use this cool command, for example:  
`python3 -mpickle dressipi*/train.txt > dressipi*/train_txt.txt`

### Train
```
python3 main.py --dataset [dressipi*]
```


## Results
My testing results:  
I also tested on SR-GNN with the same dataset, and the performances of TAGNN are indeed a little bit better.

| start month | 01     | 03     | 04     |
|-------------|--------|--------|--------|
| SR-GNN      | 11.111 | 11.528 | 11.294 |
| TAGNN       | 11.6   | 11.778 | 11.847 |


## Others
Actually, the whole TAGNN code is based on SR-GNN (the research group basically consists of same people).  
If you have any further questions, check it on the [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN) repo.
