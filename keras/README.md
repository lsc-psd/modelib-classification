# 使い方

- [訓練時](#訓練時)
- [テスト時](#テスト時)



#### 訓練時

- まずクラス数は使うモデルなどの初期設定を行ってください。初期設定ファイルはconfig.iniです。
  - model_nameには以下が利用できます。
    - VGG16, VGG19
    - InceptionV3
    - ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    - ResNeXt50
    - DenseNet121
    - MobileNetV3
  - 大文字、小文字も同じように揃えないとエラーが出るので注意してください。
- 訓練は `python train.py`にて実行可能です。
- 作ったモデルとログはcheckpoints内に格納されます。

#### テスト時

- `python test.py`にて実行してください。

