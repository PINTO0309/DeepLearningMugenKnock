# Q. 1 - 10

## Q.1. パーセプトロン

画像を読み込み、RGBをBGRの順に入れ替えよ。

画像の赤成分を取り出すには、以下のコードで可能。
cv2.imread()関数ではチャネルがBGRの順になることに注意！
これで変数redにimori.jpgの赤成分のみが入る。

```python
import cv2
img = cv2.imread("imori.jpg")
red = img[:, :, 2].copy()
```

|入力 (imori.jpg)|出力 (answer_1.jpg)|
|:---:|:---:|
|![](imori.jpg)|![](answer_1.jpg)|

答え >> [answer_1.py](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10/answer_1.py)

## Q.2. グレースケール化


