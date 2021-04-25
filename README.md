# fb-ad-block
阻擋常在facebook上出現的垃圾廣告



___

## 工具
* bs4
* selenium
* facebook-sdk
* jieba
* opencc
* tensorflow==1.14
* keras==2.25
* sklearn
* gensim
___
## 訓練
### 爬取fb貼文
需先至selenium下載對應瀏覽器
```
執行 fb_get_data.py *網址*
```
### 資料清洗
```
根據需要 執行 dataprepare.py 內函式清洗資料
```
### 訓練
採用雙向lstm+注意力模型
```
執行 train.py 
```
### 評估
觀察混淆矩陣 選擇合適閥值
```
執行 roc.py
```
## 待辦
- [ ] 利用tensorflow.js + chrome extension
攔截垃圾貼文

## 預期成果
https://youtu.be/rXfkSBuqbAQ
---
## 注意!!!
facebook可能改過
有些地方可能無法使用

## reference
w2v 字典 cna.cbow.512d.0.txt from [科技大擂台](https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1)

