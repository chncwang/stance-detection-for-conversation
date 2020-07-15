def readLines(path):
    return open(path).read().splitlines()

def readSentences(path):
    return [line[3:] for line in readLines(path)]

