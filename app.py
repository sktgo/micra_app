# 必要なモジュールのインポート
from flask import Flask
import joblib
from joblib import load
from flask import Flask, request, render_template
from wtforms import Form, StringField, SubmitField, validators
import numpy as np
import pandas as pd
from gensim.models import word2vec
import ipadic
from fugashi import GenericTagger
from sklearn.ensemble import RandomForestRegressor

w2v_model_path = "./word2vec.gensim.model"
model_path = "./model0824.joblib"


# 入力されたタイトル文字列をword2vecに変換する関数
def w2v_title(title, w2v_model):
    
    # 文章を形態素解析
    fugger = GenericTagger(ipadic.MECAB_ARGS)
    wakati_title = [w.surface for w in fugger(title)]
    # print(wakati_title)

    # モデルの語彙にない単語を無視して、各単語をベクトル化
    x = [w2v_model.wv[word] for word in wakati_title if word in w2v_model.wv]

    # 平均ベクトルを計算
    if len(x) > 0:
        title_vector = np.mean(x, axis=0)
        return title_vector
    else:
        return ValueError


# 入力された説明文字列をword2vecに変換する関数
def w2v_description(description, w2v_model):
    
    # 文章を形態素解析
    fugger = GenericTagger(ipadic.MECAB_ARGS)
    wakati_description = [w.surface for w in fugger(description)]
    # print(wakati_description)

    # モデルの語彙にない単語を無視して、各単語をベクトル化
    x = [w2v_model.wv[word] for word in wakati_description if word in w2v_model.wv]

    # 平均ベクトルを計算
    if len(x) > 0:
        description_vector = np.mean(x, axis=0)
        return description_vector
    else:
        return ValueError


# 学習済みモデルをもとに推論する関数
def predict(title, description):
    
    # W2Vモデルの読込
    w2v_model = word2vec.Word2Vec.load(w2v_model_path)

    # データフレームの作成
    # X1がtitle、X2がdescriptionをword2vecしたもの
    # w2vされた配列に角括弧[]をつけることで2次元のリストとなり、
    # pandasで横方向（列方向）に並んだDataFrameが作成される
    X1 = pd.DataFrame([w2v_title(title, w2v_model)])
    X2 = pd.DataFrame([w2v_description(description, w2v_model)])

    # Xn1 = X1.to_numpy().reshape(1, -1)
    # Xn2 = X2.to_numpy().reshape(1, -1)

    # X1 = pd.DataFrame(Xn1)
    # X2 = pd.DataFrame(Xn2)

    # 列名の作成
    columns = pd.MultiIndex.from_product([['X1'], X1.columns])
    X1.columns = columns

    columns = pd.MultiIndex.from_product([['X2'], X2.columns])
    X2.columns = columns

    # データフレームの結合
    w2v_df = pd.concat([X1, X2], axis=1)
    # print(w2v_df)
    
    # 入力変数と目的変数の指定
    # Xv = w2v_df[[f"('X1', {i})" for i in range(50)] + [f"('X2', {i})" for i in range(50)]]
    # print(Xv)
    # Xv = np.concatenate([w2v_title(title, w2v_model), w2v_description(description, w2v_model)]).reshape(1, -1)

    # 学習済みモデル（model0824.joblib）を読み込み
    model = load(model_path)
    
    # 予測値の出力
    y_pred = model.predict(w2v_df)
    return y_pred

# print(predict('あいうえお', 'かきくけこ'))

# Flask をインスタンス化
app = Flask(__name__)

# 入力フォームの設定
class MicraForm(Form):
    Title = StringField('動画タイトル文を入力してください',
                        [validators.InputRequired()])
    Description = StringField('動画の説明文を入力してください',
                              [validators.InputRequired()])

    # html側で表示するsubmitボタンの設定
    submit = SubmitField('判定')

# ルートディレクトリにアクセスがあった時の処理
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # TFormsで構築したフォームをインスタンス化
    micraform = MicraForm(request.form)
    # POSTメソッドの定義
    if request.method == 'POST':

        # 条件に当てはまらない場合
        if micraform.validate() == False:
            return render_template('index.html', forms=micraform)
        # 条件に当てはまる場合、推論を実行
        else:
            title_ = request.form['Title']
            description_ = request.form['Description']
            # 入力された文字列を使用して推論
            pred_ = int(predict(title_, description_))
            return render_template('result.html', pred = pred_)

    # GETメソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=micraform)


# アプリケーションの実行
if __name__ == '__main__':
    app.run(debug=False)
