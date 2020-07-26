# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression


def my_LogisticRegression (embeddings ,train_index, test_index, Y, seed=0,time=10):

    np.random.seed(seed)
    micro_list=[]
    macro_list=[]
    accuracy_list=[]
    precision_micro_list=[]
    precision_macro_list=[]
    recall_micro_list=[]
    recall_macro_list=[]

    Y = np.argmax(Y, -1)

    X_train = [embeddings[x] for x in train_index]
    Y_train = [Y[x] for x in train_index]
    X_test = [embeddings[x] for x in test_index]
    Y_test = [Y[x] for x in test_index]

    clf = LogisticRegression(n_jobs=40)
    if time:
        for i in range(time):
            clf.fit(X_train, Y_train)
            Y_  = clf.predict(X_test)
            micro_list.append(f1_score(Y_test, Y_, average="micro"))
            macro_list.append(f1_score(Y_test, Y_, average="macro"))
            accuracy_list.append(accuracy_score(Y_test, Y_))
            precision_micro_list.append(precision_score(Y_test, Y_, average="micro"))
            precision_macro_list.append(precision_score(Y_test, Y_, average="macro"))
            recall_micro_list.append(recall_score(Y_test, Y_, average="micro"))
            recall_macro_list.append(recall_score(Y_test, Y_, average="macro"))


        micorf1= sum(micro_list) / len(micro_list)
        macrof1 = sum(macro_list) / len(macro_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)
        # precision_micro = sum(precision_micro_list) / len(precision_micro_list)
        # precision_macro = sum(precision_macro_list) / len(precision_macro_list)
        # recall_micro = sum(recall_micro_list) / len(recall_micro_list)
        # recall_macro = sum(recall_macro_list) / len(recall_macro_list)
        # print('Micro-F1(10 avg): {:f}, Macro-F1(10avg): {:f}, accuracy(10 avg) : {:f}, Micro-pre(10 avg): {:f}, Macro-pre(10avg): {:f}, Micro-recall(10 avg) : {:f}, Macro-recall(10 avg) : {:f}'.format(micorf1, macrof1,accuracy,precision_micro,precision_macro,recall_micro,recall_macro))
        print('Micro-F1(10 avg): {:f}, Macro-F1(10avg): {:f}, accuracy(10 avg) : {:f}'.format(micorf1, macrof1,accuracy))

    return

