#!/usr/bin/env python
# coding: utf-8
# python3 vgg16_incremental.py 16 8400 1 2
# 8400, 7750

# ### imports
# ***

# In[1]:


from __future__ import print_function
import pickle, sys, time, copy
import numpy as np
from tqdm.auto import tqdm

from thundersvm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as Data
import torchvision.models as models
from torch.autograd import Variable


# ### parameters
# ***

# In[2]:


args = {}
kwargs = {}
gpu_id = 0
emb_size = 256
# es_thd = int(sys.argv[2])
n = int(sys.argv[1])
overlap = bool(int(sys.argv[2]))
n_model = 3 if overlap else 4
if n_model == 4:
    es_thd = 7750
elif n_model == 3:
    es_thd = 8400
trans_dicts = [
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0},
    {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 1},
]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using", device)

args["batch_size"] = 96
args["epochs"] = 100
args["lr"] = 0.05
args["seed"] = 5487

torch.manual_seed(args["seed"])


# ### load data for first model
# ***

# In[3]:


transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


# In[4]:


def get_cifar10():
    train_all = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
    train_x_all = torch.FloatTensor(
        [train_all.__getitem__(i)[0].tolist() for i in tqdm(range(len(train_all)))]
    )
    train_y_all = torch.LongTensor(
        [train_all.__getitem__(i)[1] for i in tqdm(range(len(train_all)))]
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("data", train=False, transform=transform_test),
        batch_size=args["batch_size"] * 2,
        shuffle=True,
        **kwargs,
    )
    return train_x_all, train_y_all, test_loader


# ### model
# ***

# In[5]:

class VGG16(nn.Module):
    def __init__(self, emb_size, n_classes=10):
        super(VGG16, self).__init__()
        self.dim = emb_size
        self.n_classes = n_classes

        backbone = models.vgg16_bn(pretrained=False)
        self.encoder = nn.Sequential(*(list(backbone.children())[:-2]))
        self.fc1 = nn.Linear(512, self.dim)

        self.cls = nn.Linear(self.dim, self.n_classes)

    def forward(self, x, return_embs=False):
        z = self.extract_emb(x)
        x = self.classify(z)

        if return_embs:
            return x, z
        return x

    def extract_emb(self, x):
        x = self.encoder(x)
        x = self.fc1(x.view(x.size(0), -1))

        return x

    def classify(self, x):
        x = self.cls(x)

        return x


# linear
class LNet(nn.Module):
    def __init__(self, emb_size, out_size):
        super(LNet, self).__init__()
        self.cls = nn.Linear(emb_size, out_size)

    def forward(self, x):
        return self.cls(x)


# MLP
class FC(nn.Module):
    def __init__(self, in_size, out_size):
        super(FC, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.linear(x))


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


# ### utils
# ***

# In[6]:


def train(epoch, model, optimizer, train_loader):
    if epoch % 15 == 0:
        print("Epoch:", epoch)
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def trainb(epoch, model1, model, optimizer):
    if epoch % 15 == 0:
        print("Epoch:", epoch)
    model1.eval()
    model.train()
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        # extract embedding of model1 and model
        with torch.no_grad():
            emb1 = model1.extract_emb(data)
        output, emb2 = model(data, return_embs=True)
        loss = criterion(output, target)
        mse_loss = mse(emb2, emb1)
        t_loss = loss + mse_loss
        t_loss.backward()
        optimizer.step()


def test(model, test_loader, task_id):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = (
                data.to(device),
                torch.LongTensor([trans_dicts[task_id][t.item()] for t in target]).to(device),
            )
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            pred = torch.LongTensor([trans_dicts[task_id][p.item()] for p in pred]).to(device)
            correct += sum((pred == target).tolist())
        print("Accuracy:", correct / len(test_loader.dataset))
    return correct


def testb(model1, model, test_loader, task_id):
    model1.eval()
    model.eval()
    mse = nn.MSELoss()
    with torch.no_grad():
        test_loss = []
        correct = 0
        for data, target in test_loader:
            data, target = (
                data.to(device),
                torch.LongTensor([trans_dicts[task_id][t.item()] for t in target]).to(device),
            )
            data, target = Variable(data), Variable(target)
            emb1 = model1.extract_emb(data)
            output, emb2 = model(data, return_embs=True)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            pred = torch.LongTensor([trans_dicts[task_id][p.item()] for p in pred]).to(device)
            correct += sum((pred == target).tolist())

            test_loss.append(mse(emb2, emb1).item())
        print(f"Accuracy: {correct / len(test_loader.dataset)}, MSELoss: {np.mean(test_loss)}")
    return correct


def test2(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            correct += sum((pred == target).tolist())
        print("Accuracy:", correct / len(test_loader.dataset))
    return correct


def extract_embedding(model, data_loader):
    ret = {"embedding": [], "target": [[] for _ in range(3)]}
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data = Variable(data.to(device))
            output = model.extract_emb(data)
            ret["embedding"] += output.tolist()
            for task_id in range(3):
                ret["target"][task_id] += [
                    trans_dicts[task_id][t] for t in target.view(-1).tolist()
                ]
    return ret


# ### train & evaluate
# ***

# In[ ]:

model_module = VGG16
scores = np.array(
    [[[[0.0 for _ in range(8)] for _ in range(n)] for _ in range(3)] for _ in range(n_model)]
)
train_x_all, train_y_all, test_loader = get_cifar10()
step_size = train_x_all.size(0) // (n_model + 1) * 2 if overlap else train_x_all.size(0) // n_model
print("step size:", step_size)

for i in range(n):
    print("round:", i)
    t0 = time.time()
    # train first model
    print("train first model")
    train_subset = Data.TensorDataset(train_x_all[:step_size], train_y_all[:step_size])
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args["batch_size"], shuffle=True, **kwargs
    )
    model1 = model_module(emb_size).to(device)
    # optimizer1 = optim.Adam(model1.parameters(), lr=args["lr"], weight_decay=5e-4)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer1, factor=0.3, patience=3, verbose=True
    # )
    optimizer1 = optim.SGD(model1.parameters(),
                           lr=args['lr'],
                           momentum=0.9,
                           weight_decay=5e-4)

    patience = es_thd
    best_acc = 0
    best_model = None

    # for epoch in range(1, args["epochs"] + 1):
    #     train(epoch, model1, optimizer1, train_loader)
    #     correct = test(model1, test_loader, task_id=0)
    #     lr_scheduler.step(-1 * correct / len(test_loader.dataset))
    #     # if correct / len(test_loader.dataset) > best_acc or correct > _th:
    #     if correct > _th:
    #         # patience = es_thd
    #         # best_acc = correct / len(test_loader.dataset)
    #         # best_model = model_module(emb_size)
    #         # best_model.load_state_dict(copy.deepcopy(model1.state_dict()))
    #         # if correct > _th:
    #         break
    #     # else:
    #         # patience -= 1
    #     # if patience <= 0:
    #     #     print("Early stopping!")
    #     #     break

    for epoch in range(1, args['epochs']+1):
        lr = args['lr'] * (0.5**(epoch//30))
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        train(epoch, model1, optimizer1, train_loader)
        correct = test(model1, test_loader, task_id=0)
        if correct >= es_thd:
            print('Early stopping!')
            break

    # restore best model
    # model1 = model_module(emb_size).to(device)
    # model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
    # model1.to(device)

    for task_id in range(3):
        scores[0][task_id][i][0] = test(model1, test_loader, task_id=task_id) / len(
            test_loader.dataset
        )
    # save data for other CLSs
    print("extract embedding")
    emb_train = extract_embedding(model1, train_loader)
    emb_test = extract_embedding(model1, test_loader)
    data = {
        "train_x": np.array(emb_train["embedding"]),
        "train_y": np.array(emb_train["target"]),
        "test_x": np.array(emb_test["embedding"]),
        "test_y": np.array(emb_test["target"]),
    }

    # train CLSs
    print("train CLSs")
    model_ls = []
    model_mlps = []
    cls_lsvms = []
    cls_svms = []
    cls_dts = []
    cls_rfs = []
    cls_lgbs = []
    out_size = {0: 10, 1: 2, 2: 3}
    lr = 1e-3
    epochs = 10

    for task_id in range(3):
        print("task:", task_id)
        # training & testing data
        train_x, train_y = (
            torch.FloatTensor(data["train_x"]),
            torch.LongTensor(data["train_y"][task_id]),
        )
        test_x, test_y = (
            torch.FloatTensor(data["test_x"]),
            torch.LongTensor(data["test_y"][task_id]),
        )
        train_loader_cls = torch.utils.data.DataLoader(
            Data.TensorDataset(train_x, train_y),
            batch_size=args["batch_size"],
            shuffle=True,
            **kwargs,
        )
        test_loader_cls = torch.utils.data.DataLoader(
            Data.TensorDataset(test_x, test_y),
            batch_size=args["batch_size"] * 2,
            shuffle=True,
            **kwargs,
        )

        # linear
        model_l = LNet(emb_size, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_l.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train(epoch, model_l, optimizer, train_loader_cls)
            correct = test2(model_l, test_loader_cls)

        scores[0][task_id][i][1] = correct / len(test_loader_cls.dataset)
        model_ls.append(model_l)

        # MLP
        model_mlp = MLP(emb_size, (emb_size + out_size[task_id]) // 2, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train(epoch, model_mlp, optimizer, train_loader_cls)
            correct = test2(model_mlp, test_loader_cls)

        scores[0][task_id][i][2] = correct / len(test_loader_cls.dataset)
        model_mlps.append(model_mlp)

        # linear svm
        cls_lsvm = SVC(kernel="linear", random_state=args["seed"], gpu_id=0)
        cls_lsvm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_lsvm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][3] = _valid_score
        cls_lsvms.append(cls_lsvm)
        print(f"Linear SVM test acc: {_valid_score:.5f}")

        # svm
        cls_svm = SVC(random_state=args["seed"], gpu_id=0)
        cls_svm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_svm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][4] = _valid_score
        cls_svms.append(cls_svm)
        print(f"SVM test acc: {_valid_score:.5f}")

        # decision tree
        cls_dt = DecisionTreeClassifier(random_state=args["seed"])
        cls_dt.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_dt.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][5] = _valid_score
        cls_dts.append(cls_dt)
        print(f"Decision Tree test acc: {_valid_score:.5f}")

        # random forest
        cls_rf = RandomForestClassifier(n_estimators=10, random_state=args["seed"], n_jobs=8)
        cls_rf.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_rf.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][6] = _valid_score
        cls_rfs.append(cls_rf)
        print(f"Random Forest test acc: {_valid_score:.5f}")

        # lgb
        cls_lgb = LGBMClassifier(random_state=args["seed"], n_jobs=8)
        cls_lgb.fit(
            data["train_x"],
            data["train_y"][task_id],
            eval_set=[(data["test_x"], data["test_y"][task_id])],
            early_stopping_rounds=100,
            verbose=100,
        )
        _valid_pred = cls_lgb.predict(data["test_x"])
        _valid_score = sum(_valid_pred == data["test_y"][task_id]) / len(_valid_pred)
        scores[0][task_id][i][7] = _valid_score
        cls_lgbs.append(cls_lgb)
        print(f"LightGBM test acc: {_valid_score:.5f}")

    # train the rest models with first model's CLS
    for j in range(1, n_model):
        print("train model:", j)
        start = j * step_size // 2 if overlap else j * step_size
        end = start + step_size
        train_subset = Data.TensorDataset(train_x_all[start:end], train_y_all[start:end])
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args["batch_size"], shuffle=True, **kwargs
        )
        model2 = model_module(emb_size).to(device)
        #         model2.cls = model1.cls
        #         model2.cls.weight.requires_grad = False
        #         model2.cls.bias.requires_grad = False
        # optimizer2 = optim.Adam(
        #     filter(lambda p: p.requires_grad, model2.parameters()), lr=args["lr"], weight_decay=5e-4
        # )
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer2, factor=0.3, patience=3, verbose=True
        # )
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, model2.parameters()),
                           lr=args['lr'],
                           momentum=0.9,
                           weight_decay=5e-4)

        patience = es_thd
        best_acc = 0
        best_model = None

        # for epoch in range(1, args["epochs"] + 1):
        #     trainb(epoch, model1, model2, optimizer2)
        #     correct = testb(model1, model2, test_loader, task_id=0)
        #     lr_scheduler.step(-1 * correct / len(test_loader.dataset))
        #     # if correct / len(test_loader.dataset) > best_acc or correct > _th:
        #     #     patience = es_thd
        #     #     best_acc = correct / len(test_loader.dataset)
        #     #     best_model = model_module(emb_size)
        #     #     best_model.load_state_dict(copy.deepcopy(model2.state_dict()))
        #     #     if correct > _th:
        #     if correct > _th:
        #         break
        #     # else:
        #     #     patience -= 1
        #     # if patience <= 0:
        #     #     print("Early stopping!")
        #     #     break

        for epoch in range(1, args['epochs'] + 51):
            lr = args['lr'] * (0.5**(epoch//30))
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr
            trainb(epoch, model1, model2, optimizer2)
            correct = testb(model1, model2, test_loader, task_id=0)
            if correct >= es_thd:
                print('Early stopping!')
                break

        # restore best model
        # model2 = model_module(emb_size)
        # model2.load_state_dict(copy.deepcopy(best_model.state_dict()))
        # model2.to(device)

        for task_id in range(3):
            scores[j][task_id][i][0] = test(model2, test_loader, task_id=task_id) / len(
                test_loader.dataset
            )
        # save data for other CLSs
        print("extract embedding")
        emb_test = extract_embedding(model2, test_loader)
        data = {"test_x": np.array(emb_test["embedding"]), "test_y": np.array(emb_test["target"])}

        # test CLSs
        print("test CLSs")
        for task_id in range(3):
            print("task:", task_id)
            # embedding
            test_x, test_y = (
                torch.FloatTensor(data["test_x"]),
                torch.LongTensor(data["test_y"][task_id]),
            )
            test_loader_cls = torch.utils.data.DataLoader(
                Data.TensorDataset(test_x, test_y),
                batch_size=args["batch_size"] * 2,
                shuffle=True,
                **kwargs,
            )

            # linear
            correct = test2(model_ls[task_id], test_loader_cls)
            scores[j][task_id][i][1] = correct / len(test_loader_cls.dataset)

            # MLP
            correct = test2(model_mlps[task_id], test_loader_cls)
            scores[j][task_id][i][2] = correct / len(test_loader_cls.dataset)

            # svm
            _valid_score = cls_lsvms[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[j][task_id][i][3] = _valid_score
            print(f"Linear SVM test acc:{_valid_score:.5f}")

            # svm
            _valid_score = cls_svms[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[j][task_id][i][4] = _valid_score
            print(f"SVM test acc:{_valid_score:.5f}")

            # decision tree
            _valid_score = cls_dts[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[j][task_id][i][5] = _valid_score
            print(f"Decision Tree test acc:{_valid_score:.5f}")

            # random forest
            _valid_score = cls_rfs[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[j][task_id][i][6] = _valid_score
            print(f"Random Forest test acc:{_valid_score:.5f}")

            # lgb
            _valid_pred = cls_lgbs[task_id].predict(data["test_x"])
            _valid_score = sum(_valid_pred == data["test_y"][task_id]) / len(_valid_pred)
            scores[j][task_id][i][7] = _valid_score
            print(f"LightGBM test acc:{_valid_score:.5f}")

    t = round(time.time() - t0)
    print("time consumed: {} min {} sec".format(t // 60, t % 60))

    file_name = (
        "scores_incremental_overlap_embedding_regularize_v4.pkl"
        if overlap
        else "scores_incremental_embedding_regularize_v4.pkl"
    )
    pickle.dump(scores, open(file_name, "wb"))

