class Data:
    def __init__(self, classes, label):
        self.classes = classes
        self.label = label


class FileSystem:
    def __init__(self):
        self.path = 'data/'

    def skip(self, f):
        firstStr = f.readline()
        secondStr = f.readline()

    def getDataFromFile(self, prefix, file):
        f = open(self.path + prefix + file, 'r')
        self.skip(f)
        data = []
        for s in f.readlines():
            data.append(list(map(int, s.split())))
        f.close()
        return Data([data[i][:-1] for i in range(len(data))], [data[i][-1] for i in range(len(data))])

    def getData(self, prefix):
        return self.getDataFromFile(prefix, '_train.txt'), self.getDataFromFile(prefix, '_test.txt')

    def writeInFile(self, fileName, message):
        f = open(fileName, 'w')
        f.write(message)
        f.close()

def getPrefixList():
    prefixes = []
    for i in range(21):
        if i + 1 < 10:
            prefixes.append('0' + str(i + 1))
        else:
            prefixes.append(str(i + 1))
    return prefixes
