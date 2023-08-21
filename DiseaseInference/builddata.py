import os
import pickle

import torch


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_data(embed_size):
    current_path = os.path.abspath(__file__)
    father_path_ = os.path.abspath(os.path.dirname(current_path))
    father_path = os.path.abspath(os.path.dirname(father_path_))
    # print(father_path)
    data_path = os.path.join(father_path, 'KGData', 'dataset_cmd')
    to_path = os.path.join(father_path, 'DiseaseInference', 'data')

    train_path = os.path.join(data_path, 'goal_cmd.pk')
    train_set = load_pickle(train_path)['train']
    test_set = load_pickle(train_path)['test']
    # dev_set = load_pickle(train_path)['dev']
    # print(train_set)
    # print(len(train_set))
    # print(len(test_set))
    # print(len(load_pickle(train_path)['all']))
    disease_ = []
    symptom_ = []
    to_train_set = []
    to_test_set = []
    to_dev_set = []
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
        update_data = {
            'SymptomEntity': [],
            # 'label': [0, 0, 0, 0, 0],
            'label': 0,
            'mask': [[0 for i in range(embed_size)] for j in range(358)]
            # 'mask': torch.zeros(1, 41)
        }
        # label
        # update_data['label'][disease_.index(data['disease_tag'])] = 1
        update_data['label'] = disease_.index(data['disease_tag'])
        # implicit_inform_slots
        for slot in list(data['implicit_inform_slots'].keys()):
            if data['implicit_inform_slots'][slot] == True:
                # update_data['mask'][symptom_.index(slot)][:] = 1
                for k in range(len(update_data['mask'][symptom_.index(slot)])):
                    update_data['mask'][symptom_.index(slot)][k] = 1
                update_data['SymptomEntity'].append(slot)
        # explicit_inform_slots
        for slot in list(data['explicit_inform_slots'].keys()):
            if data['explicit_inform_slots'][slot] == True:
                # update_data['mask'][symptom_.index(slot)][:] = 1
                for k in range(len(update_data['mask'][symptom_.index(slot)])):
                    update_data['mask'][symptom_.index(slot)][k] = 1
                update_data['SymptomEntity'].append(slot)
        # print(data)
        # print(update_data)
        to_train_set.append(update_data)

    for data in test_set:
        update_data = {
            'SymptomEntity': [],
            # 'label': [0, 0, 0, 0, 0],
            'label': 0,
            'mask': [[0 for i in range(embed_size)] for j in range(358)]
            # 'mask': torch.zeros(1, 41)
        }
        # label
        # update_data['label'][disease_.index(data['disease_tag'])] = 1
        update_data['label'] = disease_.index(data['disease_tag'])
        # implicit_inform_slots
        for slot in list(data['implicit_inform_slots'].keys()):
            if data['implicit_inform_slots'][slot] == True:
                for k in range(len(update_data['mask'][symptom_.index(slot)])):
                    update_data['mask'][symptom_.index(slot)][k] = 1
                update_data['SymptomEntity'].append(slot)
        # explicit_inform_slots
        for slot in list(data['explicit_inform_slots'].keys()):
            if data['explicit_inform_slots'][slot] == True:
                # update_data['mask'][symptom_.index(slot)][:] = 1
                for k in range(len(update_data['mask'][symptom_.index(slot)])):
                    update_data['mask'][symptom_.index(slot)][k] = 1
                update_data['SymptomEntity'].append(slot)
        # print(data)
        # print(update_data)
        to_test_set.append(update_data)

    # for data in dev_set:
    #     update_data = {
    #         'SymptomEntity': [],
    #         # 'label': [0, 0, 0, 0, 0],
    #         'label': 0,
    #         'mask': [[0 for i in range(embed_size)] for j in range(358)]
    #         # 'mask': torch.zeros(1, 41)
    #     }
    #     # label
    #     # update_data['label'][disease_.index(data['disease_tag'])] = 1
    #     update_data['label'] = disease_.index(data['disease_tag'])
    #     # implicit_inform_slots
    #     for slot in list(data['implicit_inform_slots'].keys()):
    #         if data['implicit_inform_slots'][slot] == True:
    #             for k in range(len(update_data['mask'][symptom_.index(slot)])):
    #                 update_data['mask'][symptom_.index(slot)][k] = 1
    #             update_data['SymptomEntity'].append(slot)
    #     # explicit_inform_slots
    #     for slot in list(data['explicit_inform_slots'].keys()):
    #         if data['explicit_inform_slots'][slot] == True:
    #             # update_data['mask'][symptom_.index(slot)][:] = 1
    #             for k in range(len(update_data['mask'][symptom_.index(slot)])):
    #                 update_data['mask'][symptom_.index(slot)][k] = 1
    #             update_data['SymptomEntity'].append(slot)
    #     # print(data)
    #     # print(update_data)
    #     # to_dev_set.append(update_data)
    #     to_test_set.append(update_data)
    print(len(to_train_set))
    print(len(to_test_set))
    print(len(to_dev_set))

    all_data = {}
    all_data['train'] = to_train_set
    all_data['test'] = to_test_set
    # all_data['dev'] = to_dev_set
    # print(all_data)
    with open(os.path.join(to_path, 'disease_cmd.p'), 'wb') as df:
        pickle.dump(all_data, df)  # 保存


def load_data():
    current_path = os.path.abspath(__file__)
    father_path_ = os.path.abspath(os.path.dirname(current_path))
    father_path = os.path.abspath(os.path.dirname(father_path_))
    to_path = os.path.join(father_path, 'DiseaseInference', 'data')
    data = load_pickle(os.path.join(to_path, 'disease_cmd.p'))
    # print(data)
    return data

# data_ = load_data()
# print(len(data_['train']))
# print(data_.keys())
# print(data_['test'])
# for d in data_['test']:
#     print(d['SymptomEntity'], d['label'])
build_data(100)