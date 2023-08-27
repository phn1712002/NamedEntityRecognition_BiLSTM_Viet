from Tools.H5 import loadH5

class importCSV():
    def __init__(self, path) -> None:
        self.path = path
        
    def __call__(self):

            def convert(byte):
                return str(byte.decode())

            train_dataset = loadH5(self.path + 'raw/train_dataset.h5')
            temp = list()
            for X in train_dataset.keys():
                temp.append([convert(i) for i in train_dataset[X]])
            _train_dataset = list(temp[0]), list(temp[1])

            dev_dataset = loadH5(self.path + 'raw/dev_dataset.h5')
            temp = []
            for X in dev_dataset.keys():
                temp.append([convert(i) for i in dev_dataset[X]])
            _dev_dataset = list(temp[0]), list(temp[1])

            return _train_dataset, _dev_dataset, None