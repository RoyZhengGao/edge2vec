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

def parse_args():
    parser = argparse.ArgumentParser(description="multi_lable_classification") 
    parser.add_argument('--vector-file', nargs='?', default='/Users/gaozheng/Dropbox/research/sigir_work_with_ding_ying_group/toy_example/toy_example2/toy_example_vector.txt', help='node vector')
    parser.add_argument('--label-file', nargs='?', default='/Users/gaozheng/Dropbox/research/sigir_work_with_ding_ying_group/toy_example/toy_example2/node_type.txt', help='node label')
    parser.add_argument('--clf-ratio', type=float, default=0.5,help='Input graph file')
    return parser.parse_args()

def load_word2vec_model(file):
    '''
    load node embedding model
    '''
    model = KeyedVectors.load_word2vec_format(file , binary=False)
    # print model.wv["1"]
    return model

def sample_equal_instance(ground_truth_file):
    '''
    sample equal instance from the ground truth file
    '''
    label = dict()
    with open(ground_truth_file) as f:
        for line in f:
            result = line.rstrip().split(" ")
            node = result[0]
            n_label = result[1] 
            if n_label in label:
                label[n_label].append(node)
            else:
                label_list = []
                label_list.append(node)
                label[n_label] = label_list

    min_len = 10000000
    for k,v in label.iteritems():
        random.shuffle(v)
        if len(v)<min_len:
            min_len = len(v)

    new_label = dict()
    for k,v in label.iteritems():
        new_list= []
        for i in range(min_len):
            new_list.append(v[i])
        new_label[k] = new_list

    return new_label

def data_reshape(model,label):
    '''
    reshape the data frame
    '''
    data_X = []
    data_Y = []
    for k,v in label.iteritems(): 
        for l in v: 
            data_X.append(model.wv[l])
            data_Y.append(k)
    data_X = np.asarray(data_X)
    data_Y = np.asarray(data_Y)
    return data_X,data_Y
def multi_class_classification(data_X,data_Y):
    '''
    calculate multi-class classification and return related evaluation metrics
    '''

    svc = svm.SVC(C=1, kernel='linear')
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
    # print "AUC&ROC",metrics.roc_auc_score(data_Y, predicted)
    # print "matthews_corrcoef", metrics.matthews_corrcoef(data_Y, predicted)



if __name__ == "__main__":
    args = parse_args()
    vector = load_word2vec_model(args.vector_file)
    label = sample_equal_instance(args.label_file)
    data_X, data_Y = data_reshape(vector,label)
    # print data_X,data_Y
    # print type(data_X),len(data_X)
    multi_class_classification(data_X,data_Y)







