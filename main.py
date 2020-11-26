import copy
import math
import random
import operator
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


def generateRandomDatasets(dataset):
    usedIndexes = {}
    datasets = []
    size = round(math.sqrt(len(dataset.label)))
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


def getStrToDatasetAccuracy(tree):
    blockDelimiter = '----------------------------------------------------------\n'
    return blockDelimiter + '\tDatasets index: ' + str(
        tree.idx + 1) + '\n' + '\tDepth: ' + str(tree.classifier.get_depth()) + '\n' + '\tAccuracy: ' + str(
        tree.accuracy) + '\n' + '\tCriterion: ' + tree.criterion + '\n' + \
           '\tSplitter: ' + tree.splitter + '\n' + blockDelimiter


class Solver:

    def __init__(self):
        self.prefixes = Utils.getPrefixList()
        self.fileSystem = Utils.FileSystem()
        self.criterions = ['gini', 'entropy']
        self.splitters = ['best', 'random']
        self.MAX_DEPTH = 10
        self.trees = [self.Tree(None, 0, None, None, -1) for _ in range(21)]
        self.data = []
        self.woods = []

    class Tree:
        def __init__(self, classifier, accuracy, criterion, splitter, idx):
            self.classifier = classifier
            self.accuracy = accuracy
            self.idx = idx
            self.criterion = criterion
            self.splitter = splitter

    class Wood:
        def __init__(self, trees, idx):
            self.trees = trees
            self.idx = idx

        def getAccuracy(self, dataset):
            def voting(vote):
                results = []
                for j in range(len(vote[0])):
                    answers = {}
                    for i in range(len(vote)):
                        if answers.get(vote[i][j]) is None:
                            answers[vote[i][j]] = 1
                        else:
                            answers[vote[i][j]] += 1
                    results.append(max(answers.items(), key=operator.itemgetter(1))[0])

                return results

            def getVote():
                vote = []
                for tree in self.trees:
                    vote.append(tree.predict(dataset.classes))
                return vote

            votingResult = voting(getVote())
            accuracy = 100 * sum([1 if x[0] == x[1] else 0 for x in zip(votingResult, dataset.label)]) / len(
                votingResult)
            return accuracy

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

    def printBestClassifiers(self, minClassifier, maxClassifier):
        self.fileSystem.writeInFile('MinTree.txt', getStrToDatasetAccuracy(minClassifier))
        self.fileSystem.writeInFile('MaxTree.txt', getStrToDatasetAccuracy(maxClassifier))

    def printAccuracy(self):
        f = open("treesParameters.txt", 'w')
        for tree in self.trees:
            f.write(getStrToDatasetAccuracy(tree))
        f.close()

    def setWoods(self):

        def getWood(idx, dataset):
            trees = []
            datasets = generateRandomDatasets(dataset[0])  # беру набор случайных элементов тренировочной выборки
            for dt in datasets:  # и обучаю на них лес
                criterionInd = random.randint(0, 1)
                splitterInd = random.randint(0, 1)
                tmpClassifier = DecisionTreeClassifier(criterion=self.criterions[criterionInd],
                                                       splitter=self.splitters[splitterInd])
                tmpClassifier.fit(dt.classes, dt.label)
                trees.append(tmpClassifier)
            return self.Wood(trees, idx)

        k = 0
        for i in self.data:
            self.woods.append(getWood(k, i))
            k += 1

    def printWoodAccuracyes(self, fileName, accuracyes):
        blocDelimiter = "-----------------------------------\n"
        datasetIndex = "\t Datasets index: "
        accuracy = "\t Accuracy: "
        message = ""
        for idx in range(0, len(self.woods)):
            message += str(blocDelimiter + datasetIndex + str(idx + 1) + '\n'
                           + accuracy + str(accuracyes[idx]) + '\n' + blocDelimiter)
            self.fileSystem.writeInFile(fileName, message)

    def printWoodsAccuracyes(self):
        trainAccuracys = []
        for wood in self.woods:
            trainAccuracys.append(wood.getAccuracy(self.data[wood.idx][0]))
        self.printWoodAccuracyes("WoodAndTrain.txt", trainAccuracys)
        testAccuracyes = []
        for wood in self.woods:
            testAccuracyes.append(wood.getAccuracy(self.data[wood.idx][1]))
        self.printWoodAccuracyes("WoodAndTest.txt", testAccuracyes)

    def getSolve(self):
        self.setBestTrees()
        self.printAccuracy()
        minTree, maxTree = self.findMaxAndMinDepthTrees()
        self.printBestClassifiers(minTree, maxTree)
        printGraph(minTree, self.data[minTree.idx][0], self.data[minTree.idx][1], "Min Tree", self.MAX_DEPTH)
        printGraph(maxTree, self.data[maxTree.idx][0], self.data[maxTree.idx][1], "Max Tree", self.MAX_DEPTH)
        self.setWoods()
        self.printWoodsAccuracyes()


solver = Solver()
solver.getSolve()
