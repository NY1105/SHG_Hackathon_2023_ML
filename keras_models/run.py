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

def predict(text_no=None,model_no=1):

    model = load_model(f'./checkpoint/all_attributes_model_{model_no}.h5')
    essay = load_text(order_no=text_no)
    dataObj = preprocess(essay)
    res = model.predict(dataObj.X)
    return [res, essay]

def main():
    print(predict(text_no=0,model_no=8))

if __name__ == '__main__':
    main()