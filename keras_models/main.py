# this code is authored by by Hang Jiang
import numpy as np
import glob
import os
import sys
from os import path
import tensorflow as tf
import pandas as pd
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
import model as models
from sklearn.metrics import matthews_corrcoef

from attention import AttLayer
import configuration as config
from read import read
from Preprocess import Preprocessing


    # need to update its attribute handling
def train_cross_validation(attribute, ModelName=None):

    # preprocess set up
    preprocessObj = Preprocessing()
    paramsObj = config.Params()
    preprocessObj.load_data(attribute)
    preprocessObj.load_fasttext(paramsObj.use_word_embedding)

    # set seed and cross validation (??? the value of seed)
    seed = 7
    np.random.seed(seed)
    kfolds = preprocessObj.cross_validation(preprocessObj.X, preprocessObj.onehot_Y, seed)
    cv_acc = []
    cv_records = []
    count_iter = 1
    # filepath = "./checkpoint/{}_weights.hdf5".format(preprocessObj.attribute)
    # loop the kfolds
    for train, test in kfolds:
        model_path = './checkpoint/{}_model_{}.tf'.format(preprocessObj.attribute, count_iter)
        if path.exists(model_path):
            model = load_model(model_path)
        # create objects for each fold of 10-fold CV
        else:
            # build the model
            modelObj = models.DeepModel()
            model = modelObj.chooseModel(config.ModelName, paramsObj=paramsObj, weight=preprocessObj.embedding_matrix)
        print(model.summary())

        # save the best model & history
        # filepath="weights.best.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        history = History()
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
        # checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [history, earlystop]

        # fit the model
        model.fit(preprocessObj.X[train], preprocessObj.onehot_Y[train],
                  validation_data=(preprocessObj.X[test], preprocessObj.onehot_Y[test]),
                  epochs=paramsObj.n_epoch,
                  batch_size=paramsObj.batch_size,
                  verbose=2,
                  callbacks=callbacks_list)

        # record
        model.save(model_path)
        # ct = Counter(preprocessObj.Y)
        print("working on =={}==".format(attribute))
        print("----%s: %d----" % (preprocessObj.attribute, count_iter))
        print("----highest evaluation accuracy is %f" % (100 * max(history.history['val_accuracy'])))
        # print("----dominant distribution in data is %f" % max([ct[k]*100/float(preprocessObj.Y.shape[0]) for k in ct]))
        cv_acc.append(max(history.history['val_accuracy']))
        cv_records.append(history.history['val_accuracy'])
        count_iter += 1

    outName = "./res/" + preprocessObj.attribute + ".res"
    with open(outName, 'w') as out:
        out.write(attribute.__repr__())
        out.write("\n")
        for i, score in enumerate(cv_acc):
            out.write(str(i + 1) + ': ' + str(score))
            out.write('\n')
        out.write("The Avg is %f" % np.nanmean(cv_acc))

    print(f"The {config.kfolds}-fold CV score is %s" % np.nanmean(cv_acc))


# 'attribute' can be single attribute or a list of attributes; multilabeling only used by splitting validation
def train_splitting(attribute, ModelName=None):

    # preprocess set up
    preprocessObj = Preprocessing()
    paramsObj = config.Params()
    preprocessObj.load_data(attribute)
    preprocessObj.load_fasttext(paramsObj.use_word_embedding)

    # get train, dev and test narrays
    df = pd.read_csv(config.FILENAME)
    train = df[200:].index.tolist()
    dev = df[:100].index.tolist()
    test = df[100:200].index.tolist()
    # train = df[~df.scene_id.str.split('_').str.get(1).str.contains(r'e2[0-9]')].index.tolist()
    # dev = df[df.scene_id.str.split('_').str.get(1).str.contains(r'e2[0-1]')].index.tolist()
    # test = df[df.scene_id.str.split('_').str.get(1).str.contains(r'e2[2-9]')].index.tolist()

    filepath = "./checkpoint/{}_weights.hdf5".format(preprocessObj.attribute)

    # X, Y and train, dev, test are all narray
    if path.exists(filepath):
        model = load_model(filepath)
    else:
        modelObj = models.DeepModel()
        model = modelObj.chooseModel(config.ModelName, paramsObj=paramsObj, weight=preprocessObj.embedding_matrix)
    print(model.summary())

    # save the best model & history
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    history = History()
    callbacks_list = [history, checkpoint]

    # fit the model with training data
    if config.multilabel:
        model.fit(preprocessObj.X[train], preprocessObj.onehot_Y[train],
                  # validation_data = (preprocessObj.X[dev], preprocessObj.onehot_Y[dev]),
                  epochs=paramsObj.n_epoch,
                  batch_size=paramsObj.batch_size,
                  verbose=2,
                  callbacks=callbacks_list)

        dev = np.append(dev, test)
        # predict dev set
        out = model.predict(preprocessObj.X[dev])
        out = np.array(out)

        # find the best threshold
        threshold = np.arange(0.1, 0.9, 0.1)
        acc = []
        accuracies = []
        best_threshold = np.zeros(out.shape[1])
        for i in range(out.shape[1]):
            y_prob = np.array(out[:, i])
            for j in threshold:
                y_pred = [1 if prob >= j else 0 for prob in y_prob]
                acc.append(matthews_corrcoef(preprocessObj.onehot_Y[dev][:, i], y_pred))
            acc = np.array(acc)
            index = np.where(acc == acc.max())
            accuracies.append(acc.max())
            best_threshold[i] = threshold[index[0][0]]
            acc = []

        y_pred = np.array([[1 if out[i, j] >= best_threshold[j] else 0 for j in range(preprocessObj.onehot_Y[dev].shape[1])] for i in range(len(preprocessObj.onehot_Y[dev]))])

        for idx in range(len(y_pred[0])):
            pred_col = y_pred[:, idx]
            y_col = preprocessObj.onehot_Y[dev][:, idx]
            acc = np.mean(np.equal(y_col, pred_col))
            print("accuracy is {} for {} with threshold {}".format(acc, config.dims[idx], best_threshold[idx]))

    else:
        model.fit(preprocessObj.X[train], preprocessObj.onehot_Y[train],
                  validation_data=(preprocessObj.X[dev], preprocessObj.onehot_Y[dev]),
                  epochs=paramsObj.n_epoch,
                  batch_size=paramsObj.batch_size,
                  verbose=2,
                  callbacks=callbacks_list)

        print("----highest evaluation accuracy is %f" % (100 * max(history.history['val_accuracy'])))

        # choose the best model
        best_model = load_model(filepath, custom_objects={'AttLayer': AttLayer})

        # evaluate best model
        scores = best_model.evaluate(preprocessObj.X[test], preprocessObj.onehot_Y[test], verbose=2)
        print("Test: {} loss:{} acc:{}".format(best_model.metrics, scores[0], scores[1]))

        outName = "./res/" + preprocessObj.attribute + ".res"
        with open(outName, 'w') as out:
            out.write(preprocessObj.attribute)
            out.write("\n")
            out.write('Test accuracy: {}'.format(scores[1]))
            out.write("\n")
    read()


def pure_test(attribute, ModelName=None):
    preprocessObj = Preprocessing()
    paramsObj = config.Params()
    preprocessObj.load_data(attribute)
    preprocessObj.load_fasttext(paramsObj.use_word_embedding)

    # get train, dev and test narrays
    df = pd.read_csv(config.FILENAME)
    dev = df[:100].index.tolist()
    filepath = "./checkpoint/{}_weights.hdf5".format(preprocessObj.attribute)
    model = load_model(filepath)
    out = model.predict(preprocessObj.X[dev])
    out = np.array(out)

    def check(xl):
        correct = 0
        total = len(xl)
        for i, x in enumerate(xl):
            if int(x[0]) == preprocessObj.onehot_Y[dev][i][0]:
                correct += 1
        return correct / total
    print(check(out))


def main():
    # dims = ['cAGR', 'cCON', 'cEXT', 'cNEU', 'cOPN']
    # validation
    validation_methods = [train_splitting, train_cross_validation, pure_test]
    validation_func = validation_methods[config.validation_mode]

    # model
    ModelName = config.ModelName

    # choose multiclass or multilabel
    if config.multilabel:
        dim = config.dims
        validation_func(dim, ModelName)
    else:
        dims = config.dims
        for dim in dims:
            validation_func(dim, ModelName)


if __name__ == "__main__":
    main()
