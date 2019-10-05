import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.metrics')
import gensim

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes as nb
from sklearn import svm
from sklearn.neural_network import MLPClassifier as mlpc
from sklearn.metrics import f1_score

print "Stage 0: Initiating Data Refining and Dataset Creation"
f=open('C:\\Users\\talk2\Desktop\Industrial Training ML\dev\classifier\data_1','r')
words_list_from_file=[]
a=[]

for i in range(500):
    x=f.readline()
    for a in x:
        if a in "?,./\;:()!~[]{}*<>#=-'_":
            x=x.replace(a,"")
        if a in '"':
            x=x.replace(a,"")
        if a in "&":
            x=x.replace(a," and ")
        if a in "+":
            x=x.replace(a," ")
            
    b=str(x)[1:-1].split()
    words_list_from_file.append(b)
f.close()

target=[]
f=open('C:\\Users\\talk2\Desktop\Industrial Training ML\dev\classifier\class_1','r')
for i in range(500):
    x=f.readline().split('\n')
    x.pop()
    a=str(x)[2:-2].split(" ")
    if(len(a)>2):
        target.append(a[0]) #,[a[2]]
    else:
        target.append(a[0])
f.close()
#print target
print "Stage 1: Data Loaded"

dictionary=gensim.corpora.Dictionary(words_list_from_file)
print "Stage 2: Data Dictionary Created"

bag_of_words=[dictionary.doc2bow(x) for x in words_list_from_file]
print "Stage 3: Bag of Words Made"

tokens=len(dictionary)
dense_bow=gensim.matutils.corpus2dense(bag_of_words,num_terms=tokens).transpose()
print "Stage 4: Densing Done"

tfidf=gensim.models.TfidfModel(bag_of_words)
print "Stage 5: TFIDF Model Created"

records=tfidf[bag_of_words]
dataset=gensim.matutils.corpus2dense(records,num_terms=tokens).transpose()
print "Stage 6: Dataset Created"
print "Stage 7: Initiating Classifier"


kf = KFold(n_splits = 10, shuffle = True)

accuracies = []
scores = []

for it in range(10):
    print ("Iteration ", it)
    for train, test in kf.split(dataset):
        train_set = []
        train_labels = []
        test_set = []
        test_labels = []
        for i in train:
            train_set.append(dataset[i])
            train_labels.append(target[i])
        for i in test:
            test_set.append(dataset[i])
            test_labels.append(target[i])
        
           # uncomment the classifier you want to use. Comment out the others
        
        classifier = KNeighborsClassifier()
        #classifier = nb.GaussianNB()
        #classifier = nb.MultinomialNB()
        #classifier = svm.SVC()
        #classifier = mlpc(solver = 'lbfgs', hidden_layer_sizes = (5, 15), max_iter = 200)


        predicted = classifier.fit(train_set, train_labels).predict(test_set)

        score = f1_score(test_labels, predicted, average = 'weighted')
        scores.append(score)

        incorrect = (test_labels != predicted).sum()
        accuracy = ((len(test_set) - incorrect)*100) / len(test_set)
        accuracies.append(accuracy)
        
    print("Maximum accuracy attained ", max(accuracies))
    print("f1score  ", scores[np.argmax(accuracies)])
    print('\n')

print("Maximum accuracy attained ", max(accuracies))
print("f1score  ", scores[np.argmax(accuracies)])
