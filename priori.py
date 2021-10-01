import statistics

class PrioriClassifier():
    def __init__(self, trainingY, labelToInt=None):
        self.trainingY = trainingY
        self.labelToInt = labelToInt
        self.mode = self.getMode()

    def getMode(self):
        return statistics.mode(self.trainingY)

    def getTruncatedMean(self, fraction):
        if fraction >= 0.5:
            return None
        if self.labelToInt == None:
            intTrainingY = self.trainingY
        else:
            intTrainingY = [self.labelToInt[label] for label in self.trainingY]
        
        intTrainingY.sort()
        quantiteToTruncateEachSide = int(len(intTrainingY) * fraction / 2)
        startIdx = quantiteToTruncateEachSide
        endIdx = len(intTrainingY) - quantiteToTruncateEachSide

        truncatedData = intTrainingY[startIdx:endIdx]
        truncatedMean = statistics.fmean(truncatedData)

        intToLabel = {value: label for label, value in self.labelToInt.items()}

        return intToLabel[int(round(truncatedMean))]

    def predict(self, dataX, strategy='mode', fraction=0.1):
        if strategy == 'mode':
            return [self.mode for _ in range(len(dataX))]
        if strategy == 'truncatedMean':
            truncatedMean = self.getTruncatedMean(fraction)
            return [truncatedMean for _ in range(len(dataX))]