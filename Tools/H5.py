import h5py, os

def saveH5(path, data):
    with h5py.File(path, 'w') as file:
        for key, value in data.items():
            file.create_dataset(key, data=value)
    return True

def loadH5(path):
    if os.path.exists(path):
        loaded_data = {}
        with h5py.File(path, 'r') as file:
            for key in file.keys():
                loaded_data[key] = file[key][()]
        return loaded_data
    else:
        return None

