# ディープラーニング∞本ノック!!


ディープラーニング∞本（？）ノックぅぅ

難問になるか分からないので∞本になってます。
イモリと一緒にディープラーニングの基礎からDLのライブラリの扱い、どういうDLの論文があったかを実装しながら学んでいくための問題集です。

- **問題の難易度の順番はめちゃくちゃです。なるべくポピュラーなものを採用していますが、ネタ切れであんまり聞かないものもあります笑**
- **内容はいろいろな文献を調べて載っけてるので正しくないものもあるかもしれないので注意して下さい**
- 【注意】このページを利用して、または関して生じた事に関しては、私は一切責任を負いません。すべて**自己責任**でお願い致します。


## Recent
- 2019.1.29 HSVを修正

## 環境設定

Python-3.6でやって下さい。(解答はPython-3.6で用意してます)

### 1. Minicondaのインストール

https://conda.io/miniconda.html のサイトからMinicondaをインストールします。これはWindowでもMacOSでも可能です。Minicondaがインストールできたら、端末(Windowでは端末、MacOSではターミナル)を開き、以下コマンドで仮想環境を作成します。

```bash
$ conda create python=3.6 -n gasyori100
```

作成できたら、以下コマンドで仮想環境を動作します。

```bash
$ source actiavte gasyori100
```

するとこうなります。

```bash
(gasyori100) :~/work_space/Gasyori100knock/ :$ 
```

### 2. gitのインストール

gitをインストールします。そして、端末を開いて、以下のコマンドを実行します。このコマンドでこのディレクトリを丸ごと自分のパソコンにコピーできます。

```bash
$ git clone https://github.com/yoyoyo-yo/Gasyori100knock.git
```

### 3. パッケージのインストール

以下のコマンドで必要なパッケージをインストールします。


```bash
$ pip install -r requirement.txt
```

### 4. 画像処理チュートリアル

以下のファイルを作成し sample.py という名前で保存し、実行します。

```python
import cv2

img = cv2.imread("assets/imori.jpg")
cv2.imshow("imori", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```bash
$ python sample.py
```

これで以下の画像が表示されれば成功です！
何かボタンを押せば消えます。

![](assets/sample.png)

次に画像処理に関するnumpyの扱い方のために**Tutorial**フォルダを見てみて下さい。（もう知ってるという人はスキップして下さい。）

あとは問題を解いていってください。それぞれのフォルダに問題内容が入っています。問題では assets/imori.jpg を使用して下さい。各フォルダのREADME.mdに問題、解答プログラムがあります。python answer_@@.py　とすると解答が出ます。

## 問題

詳細な問題内容は各ディレクトリのREADMEにあります。（ディレクトリで下にスクロールすればあります）
- 解答は簡易化のため、main()などは使用しません。
- numpy中心ですが、numpyの基本知識は自分で調べて下さい。


### [理論編](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10)

|番号|問題||番号|問題|
|:---:|:---:|:---:|:---:|:---:|


### [ディープラーニングをやる前の準備編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare)

|番号|問題|
|:---:|:---:|
| 1 | [データセットの読み込み](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q2-1-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) |
| 2 | [ミニバッチの作成](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q2-2-%E3%83%9F%E3%83%8B%E3%83%90%E3%83%83%E3%83%81%E4%BD%9C%E6%88%90) |
| 3 | [イテレーション・エポック](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_prepare#q2-3-%E3%82%A4%E3%83%86%E3%83%AC%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%A8%E3%82%A8%E3%83%9D%E3%83%83%E3%82%AF) |

### [フレームワークの使い方編](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/tree/master/Question_neuralnet)

|番号|問題|
|:---:|:---:|
| 1 | 共通事項 |
| 2 | [PyTorch使ったった](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_neuralnet/README_pytorch.md) |
| 3 | [Tensorflow使ったった](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_neuralnet/README_tensorflow.md) |
| 4 | [Keras使ったった](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_neuralnet/README_keras.md) |

### [画像処理編](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_31_40)

|番号|問題||番号|問題|
|:---:|:---:|:---:|:---:|:---:|



## TODO

adaptivebinalizatino, poison image blending

