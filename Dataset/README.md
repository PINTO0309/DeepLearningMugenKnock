# データセット

ディープラーニングするためのデータセット。

こんなディレクトリ構成になってる。

*seg_images* はセグメンテーションのデータ。（こちらを使用した https://github.com/wkentaro/labelme ）

```bash
train ----- images ----- akahara --- akahara_000[1-8].jpg
       |              |- madara  --- madara_000[1-8].jpg
       |
       |- seg_images --- akahara --- akahara_000[1-8].png
                      |- madara  --- madara_000[1-8].png
                      
test ----- images ----- akahara --- akahara_000[1-8].jpg
       |              |- madara  --- madara_000[1-8].jpg
       |
       |- seg_images --- akahara --- akahara_000[1-8].png
                      |- madara  --- madara_000[1-8].png

```

