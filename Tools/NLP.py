import numpy as np
from Tools.Json import saveJson, loadJson

class MapToIndex():
    def __init__(self):
        self._map = None
        self._list_key = None
        self._list_value = None


    def _createList(self, map):
        self.map = map
        self._list_key = list(map.keys())
        self._list_value = list(map.values())
        return self
    
    def settingWithList(self, listKey=None):
        map = {}
        map['UNK'] = 0
        map['PAD'] = 1
        index = 2
            
        for key in listKey:
            if not key in map.keys():
                map[key] = index
                index += 1
        
        return self._createList(map=map)
    
    def settingWithJson(self, path):
        map = loadJson(path)
        return self._createList(map=map)
    
    def settingWithDict(self, map):
        return self._createList(map=map)
        
    def encoder(self, str=None):
        if str in self._list_key:
            return self.map[str]
        else:
            return self.map['UNK']
    
    def decoder(self, index=0):
        pos = self._list_value.index(index)
        return self._list_key[pos]
    
    def getMap(self):
        return self.map
    
    def getLenMap(self):
        return len(self.map)
    
    def saveMapToJson(self, path):
        return saveJson(path, self.map)
    
    def encoderString(self, listStr=None):
        vector = []
        for i in listStr:
            vector.append(self.encoder(str(i)))
        vector = np.array(vector)
        return vector
    
    def decoderVector(self, listVector):
        listOutput = []
        for vector in listVector:
            output = []
            for index in vector:
                output.append(self.decoder(index))
            listOutput.append(output)
        return listOutput
        