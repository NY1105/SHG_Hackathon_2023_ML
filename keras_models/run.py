from .Preprocess import Preprocessing
from random import randint
import pandas as pd
# import tensorflow as tf
import keras
import glob
from collections import Counter

data_path = './data/Essays/essays2007.csv'
df = pd.read_csv(data_path)


def load_expected_result(r):
    return [df['cAGR'][r], df['cCON'][r], df['cEXT'][r], df['cOPN'][r], df['cNEU'][r]]


def load_one_essay(r):
    essay = df['text'][r]
    return essay

def load_model_all(model_path):
    model = keras.models.load_model(model_path)
    return model


def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model


def preload_model(attributes=['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU'], model_no=1):
    preloaded = []
    for attrbute in attributes:
        model_path = f'./checkpoint/{attrbute}_model_{model_no}.tf'
        preloaded.append(load_model(model_path))
    return preloaded


def load_text(order_no=0, use_dataset=True, data_path='./data/Essays/test/test.txt'):
    if use_dataset:
        return load_one_essay(r=order_no)

    with open(data_path, 'r', encoding="utf-8") as f:
        text = f.read()
        return text


def preprocess(text):
    prep = Preprocessing()
    prep.load_text(text)
    return prep


def predict(model, text_no=None):
    # model_path = f'./checkpoint/{attrbute}_model_{model_no}.h5'
    # model = load_model(model_path)
    essay = load_text(order_no=text_no)
    dataObj = preprocess(essay)
    # print(load_one_essay(text_no)[:30])
    return model.predict(dataObj.X, verbose=2)


def main():
    # attributes = ['all_attributes']
    attributes = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']
    votes = [0, 0, 0, 0, 0]
    A, C, E, O, N = preload_model(attributes=attributes, model_no=2)
    # text_number = randint(0, 2468)
    all_res = []
    no_testcases = 2468
    no_testcases = 20
    start = randint(0, 2468 - no_testcases)
    # start = 0
    text_number_desired = 2
    for text_number in range(start, start + no_testcases):
        if no_testcases == 1:
            text_number = text_number_desired
        res = []
        for attr in attributes:
            if attr == 'cAGR':
                model = A
            elif attr == 'cCON':
                model = C
            elif attr == 'cEXT':
                model = E
            elif attr == 'cOPN':
                model = O
            elif attr == 'cNEU':
                model = N
            res.append(predict(model, text_no=text_number))

        if len(attributes) == 5:
            result = []
            expected = load_expected_result(text_number)
            score = 0
            for i, attr in enumerate(attributes):
                result.append(round(res[i][0][0]))
            # print(f'result: {result}')
            # print(f'expect: {expected}')
            for i, _ in enumerate(result):
                if result[i] == expected[i]:
                    score += 1
                else:
                    votes[i] += 1
            # print(f'accuracy: {score}/5')
            all_res.append([result, expected, score])
        else:
            print(res)
    # print(load_one_essay(text_number))
    if no_testcases == 1:
        print(f'result: {result}')
        print(f'expect: {expected}')
        print(f'essay: {load_one_essay(text_number)[:100]}')
        return

    all_score = [r[2] for r in all_res]
    ct = dict(Counter(all_score))
    with open('./res/all_result.res', 'a') as f:
        f.write(f'no_testcases: {no_testcases}\n')
        f.write(f'Counter: {ct.__repr__()}\n')
        f.write(f'votes: {[f"{(int(vote/no_testcases*100))}%" for vote in votes]}\n')
        # for i,r in enumerate(all_res):
        #     f.write(f'{i}: {r}\n')


if __name__ == '__main__':
    # for i in range(5):
    main()
