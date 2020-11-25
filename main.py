import copy
import math
import random

from sklearn.tree import DecisionTreeClassifier
import Utils
import matplotlib.pyplot as plt


def getAccuracy(classifier, data):
    quantity = 0
    idx = 0
    for i in classifier.predict(data.classes):
        if data.label[idx] == i:
            quantity += 1
        idx += 1
    return 100 * quantity / len(data.classes)


def getKey(countMap):
    m = 0
    key = 0
    for k, v in countMap.items():
        if v > m:
            m = v
            key = k
    return key


def getMajorityLabel(predictions):
    predicted = []
    for _ in predictions:
        countMap = {}
        for j in range(len(predictions[0])):
            if countMap.get(j) is None:
                countMap[j] = 1
            else:
                countMap[j] += 1
        predicted.append(getKey(countMap))

    return predicted


def getWoodsAccuracy(predictions, data):
    predicted = getMajorityLabel(predictions)
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == data.label[i]:
            count += 1
    return 100 * count / len(predicted)


def getRandomDatasets(size, dataset):
    usedIndexes = {}
    datasets = []
    for i in range(round(len(dataset.label) / size)):
        indexes = []
        for q in range(size):
            j = random.randint(0, len(dataset.label) - 1)
            if j not in usedIndexes:
                indexes.append(j)
        tmpDataset = Utils.Data([], [])
        for q in indexes:
            tmpDataset.classes.append(dataset.classes[q])
            tmpDataset.label.append(dataset.label[q])
        datasets.append(copy.deepcopy(tmpDataset))

    return datasets


def printGraph(tree, trainSet, testSet, datasetName, maxDepth):
    depths = []
    trainAccuracy = []
    testAccuracy = []

    for depth in range(1, maxDepth):
        tmpClassifier = DecisionTreeClassifier(criterion=tree.criterion, splitter=tree.splitter, max_depth=depth)
        tmpClassifier.fit(trainSet.classes, trainSet.label)
        depths.append(depth)
        trainAccuracy.append(getAccuracy(tmpClassifier, trainSet))
        testAccuracy.append(getAccuracy(tmpClassifier, testSet))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set(title=datasetName,
           xlabel='Depth',
           ylabel='Accuracy')
    ax.plot(depths, trainAccuracy, color='red', label='train')
    ax.plot(depths, testAccuracy, color='blue', label='test')
    ax.legend()
    plt.show()
    fig.savefig(datasetName)


def printWoodsAccuracy(ind, accuracy, f):
    f.write('---------------------\n')
    f.write('Ind: ' + str(ind + 1) + '\n')
    f.write('Accuracy: ' + str(accuracy) + '\n')
    f.write('---------------------\n')


def printInFile(tree, f):
    f.write('----------------------------------------------------------\n')
    f.write('\tDatasets index: ' + str(tree.idx + 1) + '\n')
    f.write('\tDepth: ' + str(tree.classifier.get_depth()) + '\n')
    f.write('\tAccuracy: ' + str(tree.accuracy) + '\n')
    f.write('\tCriterion: ' + tree.criterion + '\n')
    f.write('\tSplitter: ' + tree.splitter + '\n')
    f.write('----------------------------------------------------------\n')


def printBestClassifiers(minClassifier, maxClassifier):
    f = open('MinTree.txt', 'w')
    printInFile(minClassifier, f)
    f.close()

    f = open('MaxTree.txt', 'w')
    printInFile(maxClassifier, f)
    f.close()


class Solver:

    def __init__(self):
        self.prefixes = Utils.getPrefixList()
        self.fileSystem = Utils.FileSystem()
        self.criterions = ['gini', 'entropy']
        self.splitters = ['best', 'random']
        self.MAX_DEPTH = 10
        self.trees = [self.Solve(None, 0, None, None, -1) for _ in range(21)]
        self.data = []

    class Solve:
        def __init__(self, classifier, accuracy, criterion, splitter, idx):
            self.classifier = classifier
            self.accuracy = accuracy
            self.idx = idx
            self.criterion = criterion
            self.splitter = splitter

    def setBestTrees(self):
        idx = 0
        for prefix in self.prefixes:
            datasets = self.fileSystem.getData(prefix)
            self.data.append(copy.deepcopy(datasets))
            for criterion in self.criterions:
                for splitter in self.splitters:
                    for depth in range(1, self.MAX_DEPTH + 1):
                        curClassifier = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
                        curClassifier.fit(datasets[0].classes, datasets[0].label)
                        accuracy = getAccuracy(curClassifier, datasets[1])
                        if self.trees[idx].accuracy < accuracy:
                            self.trees[idx].classifier = copy.deepcopy(curClassifier)
                            self.trees[idx].accuracy = accuracy
                            self.trees[idx].criterion = criterion
                            self.trees[idx].splitter = splitter
                            self.trees[idx].idx = idx
            idx += 1

    def findMaxAndMinDepthTrees(self):
        minTree, maxTree = copy.deepcopy(self.trees[0]), copy.deepcopy(self.trees[0])
        for tree in self.trees:
            cls = copy.deepcopy(tree.classifier)
            if cls.get_depth() > maxTree.classifier.get_depth():
                maxTree = copy.deepcopy(tree)
            if cls.get_depth() < minTree.classifier.get_depth():
                minTree = copy.deepcopy(tree)
        return minTree, maxTree

    def printAccuracy(self):
        f = open("treesParameters.txt", 'w')
        for tree in self.trees:
            printInFile(tree, f)
        f.close()

    def getWood(self, datasets):
        trees = []
        for dataset in datasets:
            criterionInd = random.randint(0, 1)
            splitterInd = random.randint(0, 1)
            tmpClassifier = DecisionTreeClassifier(criterion=self.criterions[criterionInd],
                                                   splitter=self.splitters[splitterInd])
            tmpClassifier.fit(dataset.classes, dataset.label)
            trees.append(tmpClassifier)
        return trees

    def printWood(self):
        woods = []
        lengths = []
        for dataset in self.data:
            size = round(math.sqrt(len(dataset[0].label)))
            lengths.append(size)
            datasets = getRandomDatasets(size, dataset[0])
            woods.append(self.getWood(datasets))
        ind = 0
        f = open('WoodAndTrain.txt', 'w')
        for wood in woods:
            predictions = []
            accuracy = []
            for tree in wood:
                predictions.append(tree.predict(self.data[ind][0].classes))
                accuracy.append(getWoodsAccuracy(predictions, self.data[ind][0]))
            printWoodsAccuracy(ind, accuracy, f)
            ind += 1
        f.close()

        ind = 0
        f = open('WoodAndTest.txt', 'w')
        for wood in woods:
            predictions = []
            accuracy = []
            for tree in wood:
                datasets = getRandomDatasets(lengths[ind], self.data[ind][1].classes)
                for dataset in datasets:
                    predictions.append(tree.predict(dataset))
            printWoodsAccuracy(ind, accuracy, f)
            ind += 1
        f.close()

    def getSolve(self):
        self.setBestTrees()
        self.printAccuracy()
        minTree, maxTree = self.findMaxAndMinDepthTrees()
        printBestClassifiers(minTree, maxTree)
        printGraph(minTree, self.data[minTree.idx][0], self.data[minTree.idx][1], "Min Tree", self.MAX_DEPTH)
        printGraph(maxTree, self.data[maxTree.idx][0], self.data[maxTree.idx][1], "Max Tree", self.MAX_DEPTH)
        self.printWood()


solver = Solver()
solver.getSolve()
