# 1. link prediction. negative pair and positive pair prediction
# 2. prediction result see the correlation pearson and spearman
#remember that need to down sample to equal instance
from gensim.models.keyedvectors import KeyedVectors
import argparse
import random
import numpy as np

from sklearn import linear_model                                                                                                                                               
from sklearn import metrics 
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict,train_test_split
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,mutual_info_classif,SelectFromModel 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier 
from sklearn import linear_model

import math
from scipy import stats
from scipy import spatial

def parse_args():
    parser = argparse.ArgumentParser(description="multi_lable_classification") 
    parser.add_argument('--vector-file', nargs='?', default='/Users/gaozheng/Dropbox/research/sigir_work_with_ding_ying_group/toy_example/toy_example2/toy_example_vector.txt', help='node vector')
    parser.add_argument('--pair-file', nargs='?', default='/Users/gaozheng/Dropbox/research/sigir_work_with_ding_ying_group/toy_example/toy_example2/ground_truth.txt', help='pair relation label')
    parser.add_argument('--threshold', type=float, default=0.5,help='threshold to define positive or negative')
    return parser.parse_args()


def load_word2vec_model(file):
    '''
    return node embedding model
    '''
    model = KeyedVectors.load_word2vec_format(file , binary=False)
    # print model.wv["1"]
    return model

def load_ground_truth(model,ground_truth_file,threshold): 
    '''
    load ground truth model
    '''
    true_label = []
    predicted_label = []
    with open(ground_truth_file) as f:
        for line in f:
            result = line.rstrip().split(" ")
            node1 = result[0]
            node2 = result[1] 
            relation = result[2]
            true_label.append(int(relation))
            predicted = model.wv.similarity(node1, node2)
            if predicted > threshold:
                predicted_label.append(1)
            else:
                predicted_label.append(0)
    true_label = np.asarray(true_label)
    predicted_label = np.asarray(predicted_label)
    return true_label,predicted_label

def evaluation_analysis(true_label,predicted): 
    '''
    return all metrics results
    '''
    print "accuracy",metrics.accuracy_score(true_label, predicted)
    print "f1 score macro",metrics.f1_score(true_label, predicted, average='macro')     
    print "f1 score micro",metrics.f1_score(true_label, predicted, average='micro') 
    print "precision score",metrics.precision_score(true_label, predicted, average='macro') 
    print "recall score",metrics.recall_score(true_label, predicted, average='macro') 
    print "hamming_loss",metrics.hamming_loss(true_label, predicted)
    print "classification_report", metrics.classification_report(true_label, predicted)
    print "jaccard_similarity_score", metrics.jaccard_similarity_score(true_label, predicted)
    print "log_loss", metrics.log_loss(true_label, predicted)
    print "zero_one_loss", metrics.zero_one_loss(true_label, predicted)
    print "AUC&ROC",metrics.roc_auc_score(true_label, predicted)
    print "matthews_corrcoef", metrics.matthews_corrcoef(true_label, predicted)

def correlation_analysis(v1,v2):
    '''
    calculated three correlation score between two lists
    '''
	print "spearman correlation:",str(stats.mstats.spearmanr(v1,v2).correlation)
	print "pearsonr correlation:",str(stats.mstats.pearsonr(v1,v2)[0])
	print "cosine correlation:",1 - spatial.distance.cosine(v1, v2) 


def logit_regression(model,ground_truth_file):
    '''
    return logistic regression all evaluation metrics
    '''
    label = dict()
    # true_label = []
    # instance = []
    count = 0
    with open(ground_truth_file) as f:
        for line in f:
            count = count+1
            result = line.rstrip().split(" ")
            node1 = result[0]
            node2 = result[1] 
            n_label = result[2]
            # true_label.append(int(relation))
            # vector1 = model.wv[node1]
            # vector2 = model.wv[node2]
            # vector = np.subtract(vector1, vector2) #np.concatenate((vector1,vector2), axis=0)
            if n_label in label:
                label[n_label].append(line.rstrip())
            else:
                label_list = []
                label_list.append(line.rstrip())
                label[n_label] = label_list
            # instance.append(vector)
            if count % 10000 == 0:
            # print vector,relation
                print "load data",str(count)
            
    # data_Y = np.asarray(true_label)
    # data_X = np.asarray(instance)


    #sample equal instance
    min_len = 60
    for k,v in label.iteritems():
        random.shuffle(v)
        # if len(v)<min_len:
        #     min_len = len(v)

    new_label = dict()
    for k,v in label.iteritems():
        new_list= []
        for i in range(min_len):
            new_list.append(v[i])
        new_label[k] = new_list

    #build instances
    data_X = []
    data_Y = []
    count = 0
    for k,v in new_label.iteritems(): 
        for l in v: 
            count = count + 1
            result = l.rstrip().split(" ")
            node1 = result[0]
            node2 = result[1]
            vector1 = model.wv[node1]
            vector2 = model.wv[node2]
            vector = np.concatenate((vector1,vector2), axis=0)#np.subtract(vector1, vector2) #np.concatenate((vector1,vector2), axis=0) 
            data_X.append(vector)
            data_Y.append(k)
            if count % 100 == 0:
                print "build instance",count

    data_X = np.asarray(data_X)
    data_Y = np.asarray(data_Y)
    print "min_len","len(data_Y)"
    print min_len,len(data_Y)
    # print data_X,data_Y
    # svc = svm.SVC(C=1, kernel='linear')
    svc = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

    # X_train, X_test, y_train, y_test = train_test_split( data_X, data_Y, test_size=0.4, random_state=0) 
    clf = svc.fit(data_X, data_Y) #svm
    # array = svc.coef_
    # print array
    predicted = cross_val_predict(clf, data_X, data_Y, cv=2)
    print "accuracy",metrics.accuracy_score(data_Y, predicted)
    print "f1 score macro",metrics.f1_score(data_Y, predicted, average='macro') 
    print "f1 score micro",metrics.f1_score(data_Y, predicted, average='micro') 
    print "precision score",metrics.precision_score(data_Y, predicted, average='macro') 
    print "recall score",metrics.recall_score(data_Y, predicted, average='macro') 
    print "hamming_loss",metrics.hamming_loss(data_Y, predicted)
    print "classification_report", metrics.classification_report(data_Y, predicted)
    print "jaccard_similarity_score", metrics.jaccard_similarity_score(data_Y, predicted)
    # print "log_loss", metrics.log_loss(data_Y, predicted)
    print "zero_one_loss", metrics.zero_one_loss(data_Y, predicted)
    # return true_label,predicted_label

def calculate_roc(y_true,y_pred):
    '''
    calculate AUROC score
    '''
    auroc = roc_auc_score(y_true, y_pred)
    return auroc

def plot_roc(true,pred):
    '''
    plot the ROC curve
    '''
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    plt.plot(fpr, tpr,c = "blue",markersize=2,label='edge2vec')
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    word2vec_model = load_word2vec_model(args.vector_file)
	# true_label,predicted_label = load_ground_truth(word2vec_model,args.pair_file,args.threshold)
 # 	evaluation_analysis(true_label,predicted_label)
	# correlation_analysis(true_label,predicted_label)
    logit_regression(word2vec_model,args.pair_file)




