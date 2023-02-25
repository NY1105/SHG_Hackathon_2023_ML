from main import Preprocessing
from random import randint
import pandas as pd
import tensorflow as tf
import glob

data_path = './data/Essays/essays2007.csv'


def load_one_essay(r=randint(0, 2468)):
    df = pd.read_csv(data_path)
    essay = df['text'][r]
    return essay


def load_model_all(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def load_text(order_no=None, use_dataset=True, data_path='./data/Essays/test/test.txt'):
    if use_dataset:
        if order_no:
            return load_one_essay(r=order_no)
        return load_one_essay()

    with open(data_path, 'r', encoding="utf-8") as f:
        text = f.read()
        return text


def preprocess(text):
    prep = Preprocessing()
    prep.load_text(text)
    return prep


def predict(attrbute='all_attributes', text_no=None, model_no=1):
    model_path = f'./checkpoint/{attrbute}_model_{model_no}.h5'
    model = load_model(model_path)
    essay = load_text(order_no=text_no)
    dataObj = preprocess(essay)
    return model.predict(dataObj.X)


def main():
    # attributes = ['all_attributes']
    attributes = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']
    text_number = 1
    res = []
    for attr in attributes:
        res.append(predict(attrbute=attr, text_no=text_number, model_no=1))

    print(res)
    print(load_one_essay(r=text_number))


if __name__ == '__main__':
    main()
