import torch.optim

from Agent.getdata import *
from torch.utils.data import DataLoader
from Agent.utils import *
from Agent.GAT import *
from tqdm.auto import tqdm


def train(model, data_loader, device, lr):
    model.train()
    losses = []
    for batch in tqdm(data_loader):
        label, kg_matrix, mask = [x for x in batch]
        label = label.to(device)
        kg_matrix = kg_matrix.to(device)
        mask = mask.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        output, loss = model(label, kg_matrix, mask)
        # print(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def test(model, data_loader):
    model.eval()
    losses = []
    success = 0
    total_test = 0
    for batch in tqdm(data_loader):
        total_test += 1
        # print('Test epoch:', total_test)
        with torch.no_grad():
            label, kg_matrix, mask = [x for x in batch]
            label = label.to(device)
            kg_matrix = kg_matrix.to(device)
            mask = mask.to(device)
            output, loss = model(label, kg_matrix, mask)
            # print(loss)
            losses.append(loss.item())

            success += output
    return success, total_test


def save_model(path, model, success_rate):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {}
    checkpoint['state_dict'] = model.state_dict()
    filename = 'gat_%.4f.pth.tar' % (success_rate)
    file_path = os.path.join(path, filename)
    torch.save(checkpoint, file_path)


def model_test(path_, model, data):
    model.eval()
    checkpoint = torch.load(path_)
    model.load_state_dict(checkpoint['state_dict'])
    success = 0
    total_test = 0
    for batch in tqdm(data):
        total_test += 1
        # print('Test epoch:', total_test)
        with torch.no_grad():
            label, kg_matrix, mask = [x for x in batch]
            label = label.to(device)
            kg_matrix = kg_matrix.to(device)
            mask = mask.to(device)
            output, loss = model(label, kg_matrix, mask)
            success += output
    return success, total_test


if __name__ == '__main__':
    batch_size = 32
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # print(device)
    label_num = 27
    embed_size = 100
    sym_num = 358
    lr = 0.001
    epoch_num = 2000
    dropout = 0.2
    alpha = 0.2
    heads = 2
    kg_node = 385
    dis_num = 27

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    path_ = os.path.join(father_path, 'GATModel')

    gat_model = GAT(nfeat=embed_size, nhid=embed_size, nclass=embed_size, dropout=dropout, alpha=alpha, nheads=heads, device=device, kg_node=kg_node, embed_size=embed_size, dis_num=dis_num).to(device)
    train_data = load_data()['train']
    test_data = load_data()['test']
    train_loader = DataLoader(MyDataset(train_data), batch_size, shuffle=True, collate_fn=my_collate_fn)
    # train_loader = DataLoader(MyDataset(train_data), batch_size=1, shuffle=True, collate_fn=my_collate_fn)
    test_loader = DataLoader(MyDataset(test_data), batch_size=1, shuffle=True, collate_fn=my_collate_fn)

    optimizer = torch.optim.Adam(gat_model.parameters())

    success_record = 0.
    for epoch in range(epoch_num):
        print('Train gat start !!!!!!!!!!\n')
        losses = train(gat_model, train_loader, device, lr)
        # print('Losses:', losses)
        print('Test gat start !!!!!!!!!!!\n')
        successes = 0
        total_num = 0
        success, total_test = test(gat_model, test_loader)
        successes += success
        total_num += total_test
        print('Success num: %d,Total_num: %d,Success rate: %.4f\n' % (successes, total_num, successes/total_num))
        if successes/total_num > success_record:
            print('Record best model!\n')
            success_record = successes/total_num
            save_model(path_, gat_model, success_record)
    # model_path = os.path.join(path_, 'gat_0.6677.pth.tar')
    # print(model_path)
    # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # gat_model.load_state_dict(checkpoint['state_dict'], strict=False)
    # # success, total_test = model_test(model_path, gat_model, test_loader)
    # success, total_test = test(gat_model, test_loader)
    # print('Success num: %d,Total_num: %d,Success rate: %.4f\n' % (success, total_test, success / total_test))

