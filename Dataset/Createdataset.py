from Tools.H5 import loadH5

class DatasetNERBiLSTM():
    def __init__(self, path='./Dataset/'):
        self.path = path
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        
    def __call__(self):
        
        def convert(byte):
            return str(byte.decode())
        
        train_dataset = loadH5(self.path + 'raw/train_dataset.h5')
        temp = list()
        for X in train_dataset.keys():
            temp.append([convert(i) for i in train_dataset[X]])
        self._train_dataset = list(temp[0]), list(temp[1])
        
        dev_dataset = loadH5(self.path + 'raw/dev_dataset.h5')
        temp = []
        for X in dev_dataset.keys():
            temp.append([convert(i) for i in dev_dataset[X]])
        self._dev_dataset = list(temp[0]), list(temp[1])
        
        return self._train_dataset, self._dev_dataset, self._test_dataset