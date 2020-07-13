import matplotlib.pyplot as plt


class Drawer:

    def __init__(self, datamat, result):
        self.datamat = datamat
        self.save_path = '.'

        self.models = result['models']
        self.idx = result['idx']
        self.starts = result['starts']
        self.ends = result['ends']

        self.id_dict = self._get_id_dict()

    def _get_id_dict(self):
        _dict = dict.fromkeys(set(list(self.idx)), 0)
        for key in _dict.keys():
            _dict[key] = self.idx.count(key)
        return _dict

    def vocabulary_distribution(self, semilog=False):
        data = sorted(self.id_dict.items(), key=lambda x: x[1], reverse=True)
        y = [x[1] for x in data]
        x = [x[0] for x in data]
        plt.figure()
        if semilog:
            plt.semilogy(x, y)
        else:
            plt.plot(x, y)
        plt.ylabel('vocabulary amount')
        plt.xlabel('vocabulary index')
        plt.show()
