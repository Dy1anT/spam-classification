# -*- coding: utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def MyclassifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    return p1, p0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfToken = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfToken if len(tok) > 0]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50); testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)

    X1 = []; Y1 = []; X0 = []; Y0 = []
    for docIndex in testSet:  # classify the remaining items
        if classList[docIndex] == 1:
            wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
            x1, y1 = MyclassifyNB(array(wordVector), p0V, p1V, pSpam)
            X1.append(x1)
            Y1.append(y1)
        else:
            wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
            x0, y0 = MyclassifyNB(array(wordVector), p0V, p1V, pSpam)
            X0.append(x0)
            Y0.append(y0)

    axes = plt.subplot(111)
    type0 = axes.scatter(X0, Y0, s = 100, c = 'green', alpha = 0.5)
    type1 = axes.scatter(X1, Y1, s = 100, c = 'red', alpha = 0.5)
    x = linspace(0, -600, 100)
    y = x
    axes.plot(x, y)
    plt.xlabel('p1')
    plt.ylabel('p0')
    axes.legend((type0, type1), ('ham', 'spam'), loc = 2)
    plt.show()

if __name__ == '__main__':
    spamTest()
