# Kerasを使ったった

## 1. インポート

最初は必要なものをpipでインストールする。

```bash
$ pip install tensorflow keras argparse opencv-python numpy glob
```

GPUでtensorlfowを使う場合は、tensorflow_gpuをインストールする必要がある。

コードではtensorflowをimport する。tfとエイリアスをつけることが多い。

```python
import tensorflow as tf
import keras
```

あとは必要なものももろもろimportする。

```python
import argparse
import cv2
import numpy as np
from glob import glob
```

GPU使用のときはこの記述をする。
*config.gpu_options.allow_growth = True* はGPUのメモリを必要な分だけ確保しますよーという宣言。
*config.gpu_options.visible_device_list="0"* は１個目のGPUを使いますよーっていう宣言。

```python
# GPU config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)
```

次に諸々必要な宣言をする。num_classesは分類するクラス数。今回はアカハライモリ(akahara)とマダライモリ(madara)の２クラス。img_height, img_widthは入力する画像のサイズ。

```python
num_classes = 2
img_height, img_width = 64, 64
```


## 2. モデル定義

Kerasはこのように書ける。ほとんどkerasで用意されているのでらくらｋ。

```python
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization

def Mynet():
    inputs = Input((img_height, img_width, 3))
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, name='dense1', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, name='dense2', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='model')
    return model
```

## 3. Optimizerの設定

モデルを書いたら次に最適化optimizerを設定する。
まずは定義したモデルのインスタンスを作成。

```python
model = Mynet()
```
そして肝心のoptimizerの設定ではmodel.compileという関数を使う。ここで学習率だとかモーメンタムだとか重要なハイパーパラメータを設定する。
ここではSGDで学習率0.001, モーメンタム0.9を設定。metrics=['accuracy']とすると自動的にaccuracyも計算してくれる。

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
    metrics=['accuracy'])
```

## 4. データセット用意

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
    
    xs = xs.transpose(0,3,1,2)

    return xs, ts

xs, ts = data_load('../Dataset/train/images/')
```

## 5. 学習

kerasの学習はfitなどの関数があるがここではあえて使わない。
ここからミニバッチを使って学習させる。100イテレーションを想定して、こんな感じでミニバッチを作成する。ミニバッチの作成の詳細はディープラーニング準備編を要参照。これで、xとtに学習データの入力画像、教師ラベルが格納される。

```python
mb = 8
mbi = 0
train_ind = np.arange(len(xs))
np.random.seed(0)
np.random.shuffle(train_ind)

for i in range(100):
    if mbi + mb > len(xs):
        mb_ind = train_ind[mbi:]
        np.random.shuffle(train_ind)
        mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        mbi = mb - (len(xs) - mbi)
    else:
        mb_ind = train_ind[mbi: mbi+mb]
        mbi += mb

    x = xs[mb_ind]
    t = ts[mb_ind]
```

学習では*train_on_batch*というメソッドで行える。返し値はlossとaccuracyになっている。


```python
for i in range(100):
    # syoryaku ...
    
    loss, acc = model.train_on_batch(x=x, y=t)
    print("iter >>", i+1, ",loss >>", loss, ',accuracy >>', acc)
```

## 7. 学習済みモデルの保存

モデルを学習したらそのパラメータを保存しなきゃいけない。それは*model.save()*を使う。保存名は*model.h5*とする。

```python
with tf.Session(config=config) as sess:
    for i in range(100):
        # syorayku ...
        
    model.save('model.h5')
```

以上で学習が終了!!

## 8. 学習済みモデルでテスト

次に学習したモデルを使ってテスト画像でテストする。

モデルの準備らへんはこんな感じ。*model.load_weights()* で学習済みモデルを読み込める。

```python
model = Mynet()
model.load_weights('model.h5')
````

あとはテストデータセットを読み込む。

```python
xs, ts = data_load('../Dataset/test/images/')
```

あとはテスト画像を一枚ずつモデルにフィードフォワードして予測ラベルを求めていく。これは*predict_on_batch*を使う。

```python
for x, t in zip(xs, ts):
x = np.expand_dims(x, axis=0)

    pred = model.predict_on_batch(x)[0]
    print("in {}, predicted probabilities >> {}".format(path, pred))
```

以上でtensorflowの使い方は一通り終了です。お互いおつです。


## 9. まとめたコード

以上をまとめたコードは *main_keras.py*　です。使いやすさのために少し整形してます。

学習は

```bash
$ python main_keras.py --train
```

テストは

```bash
$ python main_keras.py --test
```
