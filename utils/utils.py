import pickle


def save_pickle(a, path):
    with open(path, 'wb') as f:
        pickle.dump(a, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """

    slot_set = {}
    with open(path, 'r', encoding='utf-8') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set


def text_to_list(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """

    slot_set = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            slot_set.append(line.strip('\n').strip('\r'))
    return slot_set