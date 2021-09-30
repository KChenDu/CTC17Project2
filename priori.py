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
            fTrainingY = self.trainingY
        else:
            fTrainingY = [self.labelToInt[label] for label in self.trainingY]
        
        fTrainingY.sort()
        quantiteToTruncateEachSide = int(len(fTrainingY) * fraction / 2)
        startIdx = quantiteToTruncateEachSide
        endIdx = len(fTrainingY) - quantiteToTruncateEachSide

        truncatedData = fTrainingY[startIdx:endIdx]
        truncatedMean = statistics.fmean(truncatedData)

        intToLabel = {value: label for label, value in self.labelToInt.items()}

        return intToLabel[int(round(truncatedMean))]

    def predict(self, dataX, strategy='mode', fraction=0.1):
        if strategy == 'mode':
            return [self.mode for _ in range(len(dataX))]
        if strategy == 'truncatedMean':
            truncatedMean = self.getTruncatedMean(fraction)
            return [truncatedMean for _ in range(len(dataX))]