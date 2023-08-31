
def saveFile(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as file:
        file.write(data)
