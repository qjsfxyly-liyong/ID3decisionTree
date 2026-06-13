import operator
from math import log

##createDataSet
def createDataSet():
    """
    createDataSet
    """
    dataSet = [['C2','C1','C1','C2','B1'],
['C2','C1','C1','C1','B1'],
['C2','C1','C1','C3','B1'],
['C2','C3','C1','C1','B1'],
['C2','C2','C1','C1','B1'],
['C2','C1','C1','C3','B1'],
['C2','C2','C1','C1','B1'],
['C2','C1','C1','C2','B1'],
['C2','C1','C1','C2','B1'],
['C2','C1','C1','C2','B1'],
['C2','C2','C1','C2','B2'],
['C2','C1','C1','C3','B2'],
['C2','C2','C1','C2','B2'],
['C2','C1','C1','C3','B2'],
['C2','C2','C1','C3','B2'],
['C2','C2','C1','C2','B2'],
['C2','C1','C1','C3','B2'],
['C2','C2','C1','C3','B2'],
['C2','C2','C1','C3','B2'],
['C2','C2','C1','C1','B2'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C2','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C2','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C3','B3'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C2','B4'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C3','B4'],
['C2','C2','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C2','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C3','C1','C3','B4'],
['C2','C4','C1','C3','B4']]
    featureName = ['ideology and morality', 'cultural knowledge', 'physical psychology', 'innovative development']
    # Returns the dataset and the name of each dimension
    return dataSet, featureName

##Split dataset
def splitDataSet(dataSet,axis,value):
    """
    Divide the data set according to the given characteristics
    :param axis:Dimensions that divide the characteristics of a dataset
    :param value:Value of characteristic
    :return: All instances that meet this feature (and automatically remove this dimension feature)
    """

    # Loop through each row of data in the dataset
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis] # Delete this one-dimensional feature
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

##Calculate information entropy
#The uncertainty of the category label is always calculated
def calcShannonEnt(dataSet):
    """
    Calculate Shannon entropy of Y random variable in training data set
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet) # Number of instances
    labelCounts = {}
    for featVec in dataSet: #Traverse each instance and count the frequency of tags
        currentLabel = featVec[-1] # Represents the last column
        # If the current label is not in the labelcounts map, add labelcounts to the label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] +=1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2) # log base 2
    return shannonEnt

## Calculate conditional entropy
def calcConditionalEntropy(dataSet,i,featList,uniqueVals):
    """
Calculate the conditional entropy of Y under the given condition of X UI
: param dataset: dataset
: param i: dimension I
: param featlist: dataset feature list
: param unqiuevalues: dataset feature set
: return: conditional entropy
    """
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet,i,value)
        prob = len(subDataSet) / float(len(dataSet)) # Maximum likelihood estimation probability
        ce += prob * calcShannonEnt(subDataSet) #∑pH(Y|X=xi) Calculation of conditional entropy
    return ce

##Calculate information gain
def calcInformationGain(dataSet,baseEntropy,i):
    """
    Calculate information gain
    :param dataSet: dataSet: 
    :param baseEntropy: Information entropy of Y in dataset
    :param i: Feature dimension i
    :return: Information gain G of feature I to dataset(dataSet | X_i)
    """
    featList = [example[i] for example in dataSet] # Feature list of dimension i
    uniqueVals = set(featList) # Replace with a set, that each element in the set is not repeated
    newEntropy = calcConditionalEntropy(dataSet,i,featList,uniqueVals)#Calculate conditional entropy
    infoGain = baseEntropy - newEntropy # Information gain=information entropy - conditional entropy
    return infoGain

##Algorithm framework
def chooseBestFeatureToSplitByID3(dataSet):
    """
   Select the best dataset partition
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) -1 # The last column is classification
    baseEntropy = calcShannonEnt(dataSet) #Returns the information entropy of the entire dataset
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): # Traverse all dimension features
        infoGain = calcInformationGain(dataSet,baseEntropy,i) #Returns the information gain of a specific feature
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature # Returns the dimension corresponding to the best feature

def majorityCnt(classList):
    """
    Returns the category name with the most occurrences
    :param classList: Class list
    :retrun: The most frequent class name
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet,featureName,chooseBestFeatureToSplitFunc = chooseBestFeatureToSplitByID3):
    """
    Create decision tree
    :param dataSet: dataSet: 
    :param featureName: Name of each dimension of the dataset
    :return: decision tree
    """
    classList = [example[-1] for example in dataSet] # Category list
    if classList.count(classList[0]) == len(classList): # Count the number of column classlist[0]
        return classList[0] # When the categories are exactly the same, continue to divide
    if len(dataSet[0]) ==1: # When there is only one feature, traverse all instances to return the category with the most occurrences
        return majorityCnt(classList) # Return category label
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)#Index corresponding to the best feature
    bestFeatLabel = featureName[bestFeat] #Best features
    myTree ={bestFeatLabel:{}}  # Map structure, and key is featurelabel
    del (featureName[bestFeat])
    # Find the subset of features to classify
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = featureName[:] 
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# Construction of test decision tree
dataSet,featureName = createDataSet()
myTree = createTree(dataSet,featureName)
print(myTree)