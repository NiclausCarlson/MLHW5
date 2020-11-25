import copy

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
            self.data.append(datasets)
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

    def printGraph(self, tree, trainSet, testSet, datasetName):
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
        fig.savefig()

    def printAccuracyes(self):
        f = open("treesParameters.txt", 'w')
        for tree in self.trees:
            f.write('----------------------------------------------------------\n')
            f.write('\tDatasets index: ' + str(tree.idx + 1) + '\n')
            f.write('\tAccuracy: ' + str(tree.accuracy) + '\n')
            f.write('\tCriterion: ' + tree.criterion + '\n')
            f.write('\tSplitter: ' + tree.splitter + '\n')
            f.write('----------------------------------------------------------\n')
        f.close()

    def getSolve(self):
        self.setBestTrees()
        self.printAccuracyes()
        minTree, maxTree = self.findMaxAndMinDepthTrees()
        self.printGraph(minTree, self.data[minTree.idx][0], self.data[minTree.idx][0], "MinTreeTrainDataset")
        self.printGraph(minTree, self.data[minTree.idx][0], self.data[minTree.idx][1], "MinTreeTestDataset")
        self.printGraph(maxTree, self.data[maxTree.idx][0], self.data[maxTree.idx][0], "MaxTreeTrainDataset")
        self.printGraph(maxTree, self.data[maxTree.idx][0], self.data[maxTree.idx][1], "MaxTreeTestDataset")

        return 0


solver = Solver()
solver.getSolve()
