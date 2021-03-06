# Tensorflowを使ったった

## 1. インポート

最初は必要なものをpipでインストールする。

```bash
$ pip install tensorflow argparse opencv-python numpy glob
```

GPUでtensorlfowを使う場合は、tensorflow_gpuをインストールする必要がある。

コードではtensorflowをimport する。tfとエイリアスをつけることが多い。

```python
improt tensorflow as tf
```

あとは必要なものももろもろimportする。

```python
import argparse
import cv2
import numpy as np
from glob import glob
```

あとはこの記述が必要。これがないと変なwarningが出る。

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

次に諸々必要な宣言をする。
num_classesは分類するクラス数。今回はアカハライモリ(akahara)とマダライモリ(madara)の２クラス。
img_height, img_widthは入力する画像のサイズ。

```python
num_classes = 2
img_height, img_width = 64, 64
```


## 2. モデル定義

tensorflowでは関数としてモデルを定義できる。slimというtensorflow内のライブラリもあるが、ここではあえて自分の手でlayerを作っていく。

Convolitional layerの定義。tensorflowではtf.Variableというものを用いて自分で学習パラメータを設定する。バイアスのパラメータも同様。

```python
def conv2d(x, k=3, in_num=1, out_num=32, strides=1, activ=None, bias=True, name='conv'):
    w = tf.Variable(tf.random_normal([k, k, in_num, out_num]), name=name+'_w')
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.nn.bias_add(x, b)
    if activ is not None:
        x = activ(x)
    return x
```

プーリングの定義。

```python
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
```

Multi layer perceptronの定義。
```python
def fc(x, in_num=100, out_num=100, bias=True, activ=None, name='fc'):
    w = tf.Variable(tf.random_normal([in_num, out_num]), name=name+'_w')
    x = tf.matmul(x, w)
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.add(x, b)
    if activ is not None:
        x = activ(x)
    return x
```

これらを使ってモデルの定義は次のようにする。keep_probはdropoutの確率を制御するためのもので詳しくはあとで説明する。


```python
def Mynet(x, keep_prob):
    x = conv2d(x, k=3, in_num=3, out_num=32, activ=tf.nn.relu, name='conv1_1')
    x = conv2d(x, k=3, in_num=32, out_num=32, activ=tf.nn.relu, name='conv1_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=32, out_num=64, activ=tf.nn.relu, name='conv2_1')
    x = conv2d(x, k=3, in_num=64, out_num=64, activ=tf.nn.relu, name='conv2_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=64, out_num=128, activ=tf.nn.relu, name='conv3_1')
    x = conv2d(x, k=3, in_num=128, out_num=128, activ=tf.nn.relu, name='conv3_2')
    x = maxpool2d(x, k=2)

    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h*w*c])
    x = fc(x, in_num=w*h*c, out_num=1024, activ=tf.nn.relu, name='fc1')
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = fc(x, in_num=1024, out_num=num_classes, name='fc_out')
    return x
```

## 3. placeholder

tensorflowではplaceholderというものを使う必要がある。これは名前の通り、メモリ確保を行うためのものである。
モデルを作成する時にplaceholderで仮の入力画像みたいに扱い、メモリの場所を確保していく。keep_probはdropoutの確率を制御するためのplaceholder。

```python
X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
```

## 4. Optimizerの設定

モデルを書いたら次に最適化optimizerを設定する。
まずは定義したモデルのインスタンスを作成。

```python
logits = Mynet(X, keep_prob)
```
そして肝心のoptimizerの設定。ここで学習率だとかモーメンタムだとか重要なハイパーパラメータを設定する。
ここではAdamで学習率0.01, モーメンタム0.9を設定。

```python
preds = tf.nn.softmax(logits)
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)
```

ついでにAccuracyを計算するための仕組みもここで作成する。

```python
correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

## 5. データセット用意

あとは学習させるだけなのでデータセットを用意する。一応再掲。詳しくはディープラーニング準備編を要参照。

```bash
# get train data
def data_load(path):
    xs = np.ndarray((0, img_height, img_width, 3))
    ts = np.ndarray((0))

    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((1))
            if 'akahara' in path:
                t = np.array((0))
            elif 'madara' in path:
                t = np.array((1))
            ts = np.r_[ts, t]

    return xs, ts

xs, ts = data_load('../Dataset/train/images/')
```

## 6. 学習

tensorflowではsessionというものを使って計算を行う。
tensorflowではモデルを定義した時に入力から出力までを一つのフローとして扱い、sessionに任意の出力を指定することでその出力までを一気に計算するという方式を取る。

GPUの設定も一緒に合わせて次のように書く。このsession内で学習が行われる。
*config.gpu_options.allow_growth = True* はGPUのメモリを必要な分だけ確保しますよーっていう宣言。
*config.gpu_options.visible_device_list="0"* は一個目のGPUを使いますよーっていう宣言。

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
```

ここからミニバッチを使って学習させる。100イテレーションを想定して、こんな感じでミニバッチを作成する。ミニバッチの作成の詳細はディープラーニング準備編を要参照。これで、xとtに学習データの入力画像、教師ラベルが格納される。

```python
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb
        
        x = xs[mb_ind]
        t = ts[mb_ind]
```

次にsessionを使って勾配などを一気に計算する。
sess.run()にtrain, accuracy, lossをしているが、それぞれモデルのパラメータ更新、accuruacy計算、loss計算を行っていて、返り値もこの順になっている。feed_dict={}とはplaceholderに入力する変数を入れるもの。Xにはモデルへの入力に実際にデータセットの画像を入れる。yには教師ラベル。keep_probはdropoutの確率でここは学習なので0.5の確率でdropすることを意味する。

```python
with tf.Session(config=config) as sess:
    for i in range(100):
        # syoryaku ...

        _, acc, los = sess.run([train, accuracy, loss], feed_dict={X: x, Y: t, keep_prob: 0.5})
        
        print("iter >>", i+1, ',loss >>', los / mb, ',accuracy >>', acc)
```

## 7. 学習済みモデルの保存

モデルを学習したらそのパラメータを保存しなきゃいけない。それは*saver.save()*を使う。保存名は*cnnckptt*とする。(ckptは多分checkpointの略)

```python
with tf.Session(config=config) as sess:
    for i in range(100):
        # syorayku ...
        
    saver = tf.train.Saver()
    saver.save(sess, './cnn.ckpt')
```

以上で学習が終了!!

## 8. 学習済みモデルでテスト

次に学習したモデルを使ってテスト画像でテストする。

placeholder、モデルの準備らへんはこんな感じ。

```python
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

logits = Mynet(X, keep_prob)
out = tf.nn.softmax(logits)
````

あとはテストデータセットを読み込む。

```python
xs, ts = data_load('../Dataset/test/images/')
```

あとはテスト画像を一枚ずつモデルにフィードフォワードして予測ラベルを求めていく。これもsession内で行わなければいけない。学習済みモデルの読み込みもsession内でやる必要がある。それはsaver.restor()を使う。

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "./cnn.ckpt")

    for x, t in zip(xs, ts):
        x = np.expand_dims(x, axis=0)

        pred = out.eval(feed_dict={X: x, keep_prob: 1.0})[0]

        print("in {}, predicted probabilities >> {}".format(path, pred))
```

以上でtensorflowの使い方は一通り終了です。お互いおつです。


## 9. まとめたコード

以上をまとめたコードは *main_tensorflow.py*　です。使いやすさのために少し整形してます。

学習は

```bash
$ python main_tensorflow.py --train
```

テストは

```bash
$ python main_tensorflow.py --test
```
