from collections import defaultdict

class ZeroOutCoreCFD:
    def __init__(self, deltaUp: int, deltaDown: int, epsilon: int, source_type='VYDAJ', des_type = 'PRIJEM'):
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
        self.maxDict = {}
        self.minDict = {}
        self.source_type = source_type
        self.des_type = des_type

    def add_count(self, account, weight, sign) -> bool:
        if sign == 0: # destination acct
            self.remainDict[account] += weight
        else: # source acct
            self.remainDict[account] -= weight
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

    def __call__(self, account:int, transaction_type: str, weight: int):
        if transaction_type == 	self.des_type: #'PRIJEM' stands for Credit -- destination acct
            if self.add_count(account, weight, 0):
                raise Exception('Error!')
        elif transaction_type == self.source_type: # 'VYDAJ' stands for Debit (withdrawal) -- source acct
            if self.add_count(account, weight, 1):
                self.countDict[account] += 1
                self.countInDict[account] += self.countTempInDict[account]
                del self.countTempInDict[account]
        else: # VYBER
            # print('Unknown transaction type:', transaction_type)
            return (-1, -1)

        return (self.countDict[account], self.countInDict[account])











