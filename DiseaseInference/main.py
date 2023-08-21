import os

import torch
from DiseaseInference.builddata_infalse import *
from torch.utils.data import DataLoader
from DiseaseInference.utils import *
from DiseaseInference.DisInfer import *
from tqdm.auto import tqdm


def train(model, data_loader, device, lr):
    losses = []
    for batch in tqdm(data_loader):
        label, mask = [x for x in batch]
        label = label.to(device)
        mask = mask.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        output, loss = model(label, mask)
        # print(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def test(model, data_loader):
    losses = []
    success = 0
    total_test = 0
    for batch in tqdm(data_loader):
        total_test += 1
        # print('Test epoch:', total_test)
        with torch.no_grad():
            label, mask = [x for x in batch]
            label = label.to(device)
            mask = mask.to(device)

            output, loss = model(label, mask)
            # print(loss)
            losses.append(loss.item())

            success += output
    return success, total_test


def save_model(path, model, success_rate):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {}
    checkpoint['state_dict'] = model.state_dict()
    filename = 'disease_inference_%.4f.pth.tar' % (success_rate)
    file_path = os.path.join(path, filename)
    torch.save(checkpoint, file_path)


def model_test(path_, model, data):
    checkpoint = torch.load(path_, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    success = 0
    total_test = 0
    for batch in tqdm(data):
        total_test += 1
        # print('Test epoch:', total_test)
        with torch.no_grad():
            label, mask = [x for x in batch]
            label = label.to(device)
            mask = mask.to(device)

            ds = model.predict(mask)
            # print(label, ds)

            output, loss = model(label, mask)
            print(label, output)

            success += output
    return success, total_test


if __name__ == '__main__':
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    attention_dim = 128
    label_num = 27
    embed_size = 100
    sym_num = 358
    lr = 0.001
    epoch_num = 2000

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    path_ = os.path.join(father_path, 'DisInferModel')

    disease_model = DiseaseInference(attention_dim, label_num, embed_size, sym_num, device).to(device)

    train_data = load_data()['train']
    test_data = load_data()['test']
    print(len(test_data))
    # dev_data = load_data()['dev']
    train_loader = DataLoader(MyDataset(train_data), batch_size, shuffle=True, collate_fn=my_collate_fn)
    # train_loader = DataLoader(MyDataset(train_data), batch_size=1, shuffle=True, collate_fn=my_collate_fn)
    test_loader = DataLoader(MyDataset(test_data), batch_size=1, shuffle=False, collate_fn=my_collate_fn)

    optimizer = torch.optim.Adam(disease_model.parameters(), lr=lr)
    success_record = 0.
    # for epoch in range(epoch_num):
    #     print('Train start !!!!!!!!!!\n')
    #     losses = train(disease_model, train_loader, device, lr)
    #     # print('Losses:', losses)
    #     print('Test start !!!!!!!!!!!\n')
    #     successes = 0
    #     total_num = 0
    #     success, total_test = test(disease_model, test_loader)
    #     successes += success
    #     total_num += total_test
    #     print('Success num: %d,Total_num: %d,Success rate: %.4f\n' % (successes, total_num, successes/total_num))
    #     if successes/total_num > success_record:
    #         print('Record best model!\n')
    #         success_record = successes/total_num
    #         save_model(path_, disease_model, success_record)

    model_path = os.path.join(path_, 'disease_inference_0.6855.pth.tar')
    success, total_test = model_test(model_path, disease_model, test_loader)
    print('Success num: %d,Total_num: %d,Success rate: %.4f\n' % (success, total_test, success / total_test))






