from collections import defaultdict

class ZeroOutCore:
    def __init__(self, deltaUp: int, deltaDown: int, epsilon: int):
        self.timestamp = 0
        self.deltaUp = deltaUp
        self.deltaDown = deltaDown
        self.epsilon = epsilon
        self.remainDict = defaultdict(int)
        self.stateDict = {}
        self.countDict = defaultdict(int)
        self.countInDict = defaultdict(int)
        self.countTempInDict = defaultdict(int)
        self.countDesDict = defaultdict(int)
        self.countInDesDict = defaultdict(int)
        self.maxDict = {}
        self.minDict = {}

    def add_count(self, account, sign) -> bool:
        remain = self.remainDict[account]
        if account not in self.stateDict:
            self.stateDict[account] = None
            self.maxDict[account] = 0
            self.minDict[account] = 0
        oldState = self.stateDict[account]

        if sign == 0:  # sign: 0 -> as destination acc
            if (remain - self.minDict[account]) > self.deltaUp:
                self.stateDict[account] = 1  # up
                self.countTempInDict[account] += 1
        elif sign == 1: # sign: 1 -> as source acc
            if (self.maxDict[account] - remain) > self.deltaDown and remain <= (self.minDict[account]+self.epsilon):
                self.stateDict[account] = -1 # down

        if remain > self.maxDict[account]:
            self.maxDict[account] = remain
        if remain < self.minDict[account]:
            self.minDict[account] = remain

        if oldState == 1 and self.stateDict[account] == -1:
            self.minDict[account] = remain
            return True
        if oldState == -1 and self.stateDict[account] == 1:
            self.maxDict[account] = remain
        return False

    def __call__(self, source: int, destination:int, timestamp: int, weight: int):
        self.remainDict[source] -= weight
        self.remainDict[destination] += weight

        if self.add_count(source, 1):
            self.countDict[source] += 1
            self.countInDict[source] += self.countTempInDict[source]
            self.countDesDict[destination] += 1
            self.countInDesDict[destination] += self.countTempInDict[source]
            del self.countTempInDict[source]

        if self.add_count(destination, 0):
            raise Exception('Error!')

        return ((self.countDict[source], self.countInDict[source], self.countDesDict[source], self.countInDesDict[source]), \
                (self.countDict[destination], self.countInDict[destination], self.countDesDict[destination], self.countInDesDict[destination]))












