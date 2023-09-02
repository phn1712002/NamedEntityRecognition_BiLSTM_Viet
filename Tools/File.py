import json, os
def saveFile(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as json_file:
        json.dump(data, json_file)

def loadFile(path, encoding='utf-8'):
    if os.path.exists(path):
        with open(path, 'r', encoding=encoding) as json_file:
            return json.load(json_file)
    else:
        return None