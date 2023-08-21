import copy
import os
import pickle

import torch
from KGData.KG import KGADJ
import dialog_config


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


entity_num = 385
kg = KGADJ(dialog_config.device)
current_path = os.path.abspath(__file__)
father_path_ = os.path.abspath(os.path.dirname(current_path))
father_path = os.path.abspath(os.path.dirname(father_path_))
# print(father_path)
data_path = os.path.join(father_path, 'KGData', 'dataset_cmd')
to_path = os.path.join(father_path, 'Agent', 'GAT-Data', 'data')

train_path = os.path.join(data_path, 'goal_cmd.pk')
train_set = load_pickle(train_path)['train']
test_set = load_pickle(train_path)['test']
slot_set = text_to_dict(os.path.join(data_path, 'cmd_slot_set.txt'))  # all slots with symptoms + all disease
# print(slot_set)
# print(train_set)
# print(len(train_set))
# print(len(test_set))
# print(len(load_pickle(train_path)['all']))
disease_ = []
symptom_ = []
to_train_set = []
to_test_set = []
with open(os.path.join(data_path, 'diseases_cmd.txt'), 'r', encoding='utf-8') as d:
    content = d.readlines()
    for line in content:
        info = line.strip().split('\t')
        disease_.append(info[0])
# print(disease_)

with open(os.path.join(data_path, 'symptoms_cmd.txt'), 'r', encoding='utf-8') as d:
    content = d.readlines()
    for line in content:
        info = line.strip().split('\t')
        symptom_.append(info[0])
# print(symptom_)

for data in train_set:
    kg.initialize_adj()
    matrix_ = copy.deepcopy(kg.kg_matrix)
    current_confirm_sym = []
    mask = torch.zeros(entity_num).reshape(-1, entity_num)
    update_data = {
        'SymptomEntity': [],
        'label': 0,
        'kg_matrix': matrix_,
        'mask': mask
    }
    # label
    update_data['label'] = disease_.index(data['disease_tag'])
    # implicit_inform_slots
    for slot in list(data['implicit_inform_slots'].keys()):
        if data['implicit_inform_slots'][slot] == True:
            current_confirm_sym.append(slot)
            update_data['SymptomEntity'].append(slot)
            update_data['mask'][0][slot_set[slot]] = dialog_config.TRUE
    # explicit_inform_slots
    for slot in list(data['explicit_inform_slots'].keys()):
        if data['explicit_inform_slots'][slot] == True:
            current_confirm_sym.append(slot)
            update_data['SymptomEntity'].append(slot)
            update_data['mask'][0][slot_set[slot]] = dialog_config.TRUE
    kg.update_adj(current_confirm_sym)
    update_data['kg_matrix'] = copy.deepcopy(kg.kg_matrix)
    # print(data)
    # print(update_data)
    to_train_set.append(update_data)

for data in test_set:
    kg.initialize_adj()
    matrix_ = copy.deepcopy(kg.kg_matrix)
    current_confirm_sym = []
    mask = torch.zeros(entity_num).reshape(-1, entity_num)
    update_data = {
        'SymptomEntity': [],
        'label': 0,
        'kg_matrix': matrix_,
        'mask': mask
    }
    # label
    update_data['label'] = disease_.index(data['disease_tag'])
    # implicit_inform_slots
    for slot in list(data['implicit_inform_slots'].keys()):
        if data['implicit_inform_slots'][slot] == True:
            current_confirm_sym.append(slot)
            update_data['SymptomEntity'].append(slot)
            update_data['mask'][0][slot_set[slot]] = dialog_config.TRUE
    # explicit_inform_slots
    for slot in list(data['explicit_inform_slots'].keys()):
        if data['explicit_inform_slots'][slot] == True:
            current_confirm_sym.append(slot)
            update_data['SymptomEntity'].append(slot)
            update_data['mask'][0][slot_set[slot]] = dialog_config.TRUE
    kg.update_adj(current_confirm_sym)
    update_data['kg_matrix'] = copy.deepcopy(kg.kg_matrix)
    # print(data)
    # print(update_data)
    to_test_set.append(update_data)

# print(len(to_train_set))
# print(len(to_test_set))
all_data = {}
all_data['train'] = to_train_set
all_data['test'] = to_test_set
print(len(all_data['train']))
print(len(all_data['test']))
print(all_data)
with open(os.path.join(to_path, 'disease_cmd.p'), 'wb') as df:
    pickle.dump(all_data, df)  # 保存

