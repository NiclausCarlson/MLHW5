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


def getWoodsAccuracy(answers, labels):

    return 0


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


def printGraph(tree, trainSet, testSet, datasetName):
    depths = []
    accuracy = []
    for depth in range(1, tree.classifier.get_depth() + 1):
        tmpClassifier = DecisionTreeClassifier(criterion=tree.criterion, splitter=tree.splitter, max_depth=depth)
        tmpClassifier.fit(trainSet.classes, trainSet.label)
        depths.append(depth)
        accuracy.append(getAccuracy(tmpClassifier, testSet))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set(title=datasetName,
           xlabel='Depth',
           ylabel='Accuracy')
    ax.plot(depths, accuracy, color='red')
    plt.show()
    fig.savefig(datasetName)


def printWoodsAccuracy(ind, accuracy, f):
    f.write('---------------------\n')
    f.write('Ind: ' + str(ind + 1) + '\n')
    f.wrinte('Accuracy: ' + str(accuracy) + '\n')
    f.write('---------------------\n')


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
        minTree, maxTree = self.trees[0], self.trees[0]
        for tree in self.trees:
            cls = tree.classifier
            if cls.get_depth() > maxTree.classifier.get_depth():
                maxTree = tree
            if cls.get_depth() < minTree.classifier.get_depth():
                minTree = tree
        return minTree, maxTree

    def printAccuracy(self):
        f = open("treesParameters.txt", 'w')
        for tree in self.trees:
            f.write('----------------------------------------------------------\n')
            f.write('\tDatasets index: ' + str(tree.idx + 1) + '\n')
            f.write('\tDepth: ' + str(tree.classifier.get_depth()) + '\n')
            f.write('\tAccuracy: ' + str(tree.accuracy) + '\n')
            f.write('\tCriterion: ' + tree.criterion + '\n')
            f.write('\tSplitter: ' + tree.splitter + '\n')
            f.write('----------------------------------------------------------\n')
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
        trainResults = []
        ind = 0
        f = open('WoodAndTrain.txt', 'w')
        for wood in woods:
            for tree in wood:
                trainResults.append(tree.predict(self.data[ind][0].classes))
        f.close()
        testsResults = []
        f = open('WoodAndTest.txt', 'w')
        for wood in woods:
            for tree in wood:
                datasets = getRandomDatasets(lengths[ind], self.data[ind][1].classes)
                for dataset in datasets:
                    testsResults.append(tree.predict(dataset))
        f.close()

    def getSolve(self):
        self.setBestTrees()
        # self.printAccuracy()
        # minTree, maxTree = self.findMaxAndMinDepthTrees()
        # self.printGraph(minTree, self.data[minTree.idx][0], self.data[minTree.idx][0], "Min Tree Train Dataset")
        # self.printGraph(minTree, self.data[minTree.idx][0], self.data[minTree.idx][1], "Min Tree Test Dataset")
        # self.printGraph(maxTree, self.data[maxTree.idx][0], self.data[maxTree.idx][0], "Max Tree Train Dataset")
        # self.printGraph(maxTree, self.data[maxTree.idx][0], self.data[maxTree.idx][1], "Max Tree Test Dataset")
        self.printWood()


solver = Solver()
solver.getSolve()
