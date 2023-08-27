from Dataset.importTXT import importTXT
from Dataset.importCSV import importCSV

class DatasetNERBiLSTM():
    def __init__(self, path='./Dataset/'):
        self.path = path
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        
    def __call__(self):
        #self._train_dataset = importTXT(path=self.path + 'raw/')()
        self._train_dataset, self._dev_dataset, self._test_dataset = importCSV(path=self.path)()
        return self._train_dataset, self._dev_dataset, self._test_dataset