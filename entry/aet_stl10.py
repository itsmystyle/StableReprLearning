#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import pickle
import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from tqdm.auto import tqdm
from torch.autograd import Variable
from torchvision import datasets, transforms
# from thundersvm import SVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

args = {}
kwargs = {}
emb_size = 256
es_thd = int(sys.argv[2])
n = int(sys.argv[1])

# STL10
# ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
trans_dicts = [
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
    {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 1},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

args["batch_size"] = 32
args["epochs"] = 100
args["lr"] = 1e-3
args["seed"] = 4896

args["base_dir"] = "model/aet_stl10/nofix"
args["model_save_path"] = "model_{}_{}.pkl"
args["train_save_path"] = "train_{}_{}.pkl"
args["test_save_path"] = "test_{}_{}.pkl"

args["perceptron_save_path"] = "perceptron_{}_{}.pkl"
args["mlp_save_path"] = "mlp_{}_{}.pkl"
args["lsvm_save_path"] = "lsvm_{}_{}.pkl"
args["svm_save_path"] = "svm_{}_{}.pkl"
args["dt_save_path"] = "dt_{}_{}.pkl"
args["rf_save_path"] = "rf_{}_{}.pkl"

torch.manual_seed(args["seed"])

transform_train = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])


def get_stl10():
    train_loader = torch.utils.data.DataLoader(
        datasets.STL10("data", split="train", download=True, transform=transform_test),
        batch_size=args["batch_size"],
        num_workers=8,
        shuffle=False,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.STL10("data", split="test", transform=transform_test),
        batch_size=args["batch_size"] * 4,
        num_workers=8,
        shuffle=False,
        **kwargs,
    )
    unlabel_loader = torch.utils.data.DataLoader(
        datasets.STL10("data", split="unlabeled", transform=transform_train),
        batch_size=args["batch_size"],
        num_workers=8,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader, unlabel_loader


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Autoencoder(nn.Module):
    def __init__(self, emb_size):
        super(Autoencoder, self).__init__()
        self.dim = emb_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(2048, self.dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.dim, 2048),
            Reshape(-1, 128, 4, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
        )

    def forward(self, x, return_embs=False):
        z = self.extract_emb(x)
        x = self.decoding(z)

        if return_embs:
            return x, z
        return x

    def extract_emb(self, x):
        x = self.encoder(x)
        x = self.fc1(x.view(x.size(0), -1))

        return x

    def decoding(self, x):
        x = self.decoder(x)

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


def imsave(img, path):
    torchvision.utils.save_image(img.cpu(), path)


def ae_train(epoch, model, optimizer):
    print("Epoch:", epoch)
    model.train()
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    loss_ls = []
    for batch_idx, (data, _) in enumerate(tqdm(stl10_unlabel_loader)):
        data = data.to(device)
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        loss_ls.append(loss.item())

    return np.mean(loss_ls)


def train(epoch, model, optimizer):
    print("Epoch:", epoch)
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def ae_test(epoch, model, retrain=False):
    model.eval()
    criterion = nn.MSELoss()
    loss_ls = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(stl10_train_loader)):
            data = data.to(device)
            data = Variable(data)
            output = model(data)
            loss = criterion(output, data)

            loss_ls.append(loss.item())

    return np.mean(loss_ls)


def test2(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in tqdm(test_loader):
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
        for data, target in tqdm(data_loader):
            data = Variable(data.to(device))
            output = model.extract_emb(data)
            ret["embedding"] += output.tolist()
            for task_id in range(3):
                ret["target"][task_id] += [
                    trans_dicts[task_id][t] for t in target.view(-1).tolist()
                ]
    return ret


model_module = Autoencoder
scores = np.array(
    [[[[0.0 for _ in range(7)] for _ in range(n)] for _ in range(3)] for _ in range(2)]
)

for i in range(n):
    print("round:", i)
    t0 = time.time()

    Path(args["base_dir"] + f"/{i}").mkdir(parents=True, exist_ok=True)

    # train first model
    print("train first model")
    # cifar10_train_loader, cifar10_test_loader = get_cifar10()
    stl10_train_loader, stl10_test_loader, stl10_unlabel_loader = get_stl10()
    model1 = model_module(emb_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args["lr"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, factor=0.3, patience=3, verbose=True
    )

    patience = es_thd
    best_acc = 1000000000.0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        train_loss = ae_train(epoch, model1, optimizer1)
        test_loss = ae_test(epoch, model1)
        lr_scheduler.step(test_loss)
        print(f"train loss: {train_loss:.5f}, test loss: {test_loss:.5f}")

        if test_loss < best_acc:
            patience = es_thd
            best_acc = test_loss
            best_model = model_module(emb_size)
            best_model.load_state_dict(copy.deepcopy(model1.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model1 = model_module(emb_size)
    model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model1.to(device)

    # save model
    save_model_path = args["base_dir"] + f"/{i}/" + args["model_save_path"].format(0, 1)
    torch.save(model1.state_dict(), save_model_path)

    # save data for other CLSs
    print("extract embedding")
    emb_train = extract_embedding(model1, stl10_train_loader)
    emb_test = extract_embedding(model1, stl10_test_loader)
    data = {
        "train_x": np.array(emb_train["embedding"]),
        "train_y": np.array(emb_train["target"]),
        "test_x": np.array(emb_test["embedding"]),
        "test_y": np.array(emb_test["target"]),
    }

    save_model_path = args["base_dir"] + f"/{i}/" + args["train_save_path"].format(0, 1)
    with open(save_model_path, "wb") as fout:
        _dict = {"x": data["train_x"], "y": data["train_y"]}
        pickle.dump(_dict, fout)

    save_model_path = args["base_dir"] + f"/{i}/" + args["test_save_path"].format(0, 1)
    with open(save_model_path, "wb") as fout:
        _dict = {"x": data["test_x"], "y": data["test_y"]}
        pickle.dump(_dict, fout)

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
    epochs = 50

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
        train_loader = torch.utils.data.DataLoader(
            Data.TensorDataset(train_x, train_y),
            batch_size=args["batch_size"],
            num_workers=8,
            shuffle=True,
            **kwargs,
        )
        test_loader = torch.utils.data.DataLoader(
            Data.TensorDataset(test_x, test_y),
            batch_size=args["batch_size"] * 4,
            num_workers=8,
            shuffle=False,
            **kwargs,
        )

        # linear
        model_l = LNet(emb_size, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_l.parameters(), lr=lr)
        patience = es_thd
        best_acc = 0
        best_model = None

        for epoch in range(1, epochs + 1):
            train(epoch, model_l, optimizer)
            correct = test2(model_l)
            if correct / len(test_loader.dataset) > best_acc:
                patience = es_thd
                best_acc = correct / len(test_loader.dataset)
                best_model = LNet(emb_size, out_size[task_id])
                best_model.load_state_dict(copy.deepcopy(model_l.state_dict()))
            else:
                patience -= 1
            if patience <= 0:
                break

        # restore best model
        model_l = LNet(emb_size, out_size[task_id])
        model_l.load_state_dict(copy.deepcopy(best_model.state_dict()))
        model_l.to(device)

        # save model
        save_model_path = (
            args["base_dir"] + f"/{i}/" + args["perceptron_save_path"].format(task_id, 1)
        )
        torch.save(model_l.state_dict(), save_model_path)

        scores[0][task_id][i][0] = correct / len(test_loader.dataset)
        model_ls.append(model_l)

        # MLP
        model_mlp = MLP(emb_size, (emb_size + out_size[task_id]) // 2, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=lr)
        patience = es_thd
        best_acc = 0
        best_model = None

        for epoch in range(1, epochs + 1):
            train(epoch, model_mlp, optimizer)
            correct = test2(model_mlp)
            if correct / len(test_loader.dataset) > best_acc:
                patience = es_thd
                best_acc = correct / len(test_loader.dataset)
                best_model = MLP(emb_size, (emb_size + out_size[task_id]) // 2, out_size[task_id])
                best_model.load_state_dict(copy.deepcopy(model_mlp.state_dict()))
            else:
                patience -= 1
            if patience <= 0:
                break

        # restore best model
        model_mlp = MLP(emb_size, (emb_size + out_size[task_id]) // 2, out_size[task_id])
        model_mlp.load_state_dict(copy.deepcopy(best_model.state_dict()))
        model_mlp.to(device)

        # save model
        save_model_path = args["base_dir"] + f"/{i}/" + args["mlp_save_path"].format(task_id, 1)
        torch.save(model_mlp.state_dict(), save_model_path)

        scores[0][task_id][i][1] = correct / len(test_loader.dataset)
        model_mlps.append(model_mlp)

        # linear svm
        cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
        cls_lsvm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_lsvm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][2] = _valid_score
        cls_lsvms.append(cls_lsvm)
        print(f"Linear SVM test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["lsvm_save_path"].format(task_id, 1)
        with open(save_model_path, "wb") as fout:
            pickle.dump(cls_lsvm, fout)

        # svm
        cls_svm = SVC(random_state=args["seed"])
        cls_svm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_svm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][3] = _valid_score
        cls_svms.append(cls_svm)
        print(f"SVM test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["svm_save_path"].format(task_id, 1)
        with open(save_model_path, "wb") as fout:
            pickle.dump(cls_svm, fout)

        # decision tree
        cls_dt = DecisionTreeClassifier(random_state=args["seed"])
        cls_dt.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_dt.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][4] = _valid_score
        cls_dts.append(cls_dt)
        print(f"Decision Tree test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["dt_save_path"].format(task_id, 1)
        with open(save_model_path, "wb") as fout:
            pickle.dump(cls_dt, fout)

        # random forest
        cls_rf = RandomForestClassifier(n_estimators=10, random_state=args["seed"], n_jobs=16)
        cls_rf.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_rf.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][5] = _valid_score
        cls_rfs.append(cls_rf)
        print(f"Random Forest test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["rf_save_path"].format(task_id, 1)
        with open(save_model_path, "wb") as fout:
            pickle.dump(cls_rf, fout)

        # lgb
        cls_lgb = LGBMClassifier(random_state=args["seed"], n_jobs=16)
        cls_lgb.fit(
            data["train_x"],
            data["train_y"][task_id],
            eval_set=[(data["test_x"], data["test_y"][task_id])],
            early_stopping_rounds=100,
            verbose=100,
        )
        _valid_pred = cls_lgb.predict(data["test_x"])
        _valid_score = sum(_valid_pred == data["test_y"][task_id]) / len(_valid_pred)
        scores[0][task_id][i][6] = _valid_score
        cls_lgbs.append(cls_lgb)
        print(f"LightGBM test acc: {_valid_score:.5f}")

    # train second model with first model's CLS
    print("train second model")
    stl10_train_loader, stl10_test_loader, stl10_unlabel_loader = get_stl10()
    model2 = model_module(emb_size).to(device)
    # model2.decoder = model1.decoder
    # for param in model2.decoder.parameters():
    #     param.requires_grad = False
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=args["lr"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, factor=0.3, patience=3, verbose=True
    )

    patience = es_thd
    best_acc = 1000000000.0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        train_loss = ae_train(epoch, model2, optimizer2)
        test_loss = ae_test(epoch, model2, True)
        lr_scheduler.step(test_loss)
        print(f"train loss: {train_loss:.5f}, test loss: {test_loss:.5f}")

        if test_loss < best_acc:
            patience = es_thd
            best_acc = test_loss
            best_model = model_module(emb_size)
            best_model.load_state_dict(copy.deepcopy(model2.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model2 = model_module(emb_size)
    model2.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model2.to(device)

    # save model
    save_model_path = args["base_dir"] + f"/{i}/" + args["model_save_path"].format(0, 2)
    torch.save(model2.state_dict(), save_model_path)

    # save data for other CLSs
    print("extract embedding")
    emb_train = extract_embedding(model1, stl10_train_loader)
    emb_test = extract_embedding(model1, stl10_test_loader)
    data = {
        "train_x": np.array(emb_train["embedding"]),
        "train_y": np.array(emb_train["target"]),
        "test_x": np.array(emb_test["embedding"]),
        "test_y": np.array(emb_test["target"]),
    }

    save_model_path = args["base_dir"] + f"/{i}/" + args["train_save_path"].format(0, 2)
    with open(save_model_path, "wb") as fout:
        _dict = {"x": data["train_x"], "y": data["train_y"]}
        pickle.dump(_dict, fout)

    save_model_path = args["base_dir"] + f"/{i}/" + args["test_save_path"].format(0, 2)
    with open(save_model_path, "wb") as fout:
        _dict = {"x": data["test_x"], "y": data["test_y"]}
        pickle.dump(_dict, fout)

    # test CLSs
    print("test CLSs")
    for task_id in range(3):
        print("task:", task_id)
        # embedding
        test_x, test_y = (
            torch.FloatTensor(data["test_x"]),
            torch.LongTensor(data["test_y"][task_id]),
        )
        test_loader = torch.utils.data.DataLoader(
            Data.TensorDataset(test_x, test_y),
            batch_size=args["batch_size"] * 2,
            shuffle=False,
            **kwargs,
        )

        # linear
        correct = test2(model_ls[task_id])
        scores[1][task_id][i][0] = correct / len(test_loader.dataset)

        # MLP
        correct = test2(model_mlps[task_id])
        scores[1][task_id][i][1] = correct / len(test_loader.dataset)

        # svm
        _valid_score = cls_lsvms[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][2] = _valid_score
        print(f"Linear SVM test acc:{_valid_score:.5f}")

        # svm
        _valid_score = cls_svms[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][3] = _valid_score
        print(f"SVM test acc:{_valid_score:.5f}")

        # decision tree
        _valid_score = cls_dts[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][4] = _valid_score
        print(f"Decision Tree test acc:{_valid_score:.5f}")

        # random forest
        _valid_score = cls_rfs[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][5] = _valid_score
        print(f"Random Forest test acc:{_valid_score:.5f}")

        # lgb
        _valid_pred = cls_lgbs[task_id].predict(data["test_x"])
        _valid_score = sum(_valid_pred == data["test_y"][task_id]) / len(_valid_pred)
        scores[1][task_id][i][6] = _valid_score
        print(f"LightGBM test acc:{_valid_score:.5f}")

    t = round(time.time() - t0)
    print("time consumed: {} min {} sec".format(t // 60, t % 60))

pickle.dump(
    scores,
    open("results/scores_autoencoder_d256_stl10_nofix_decoder_10_save_model.pkl", "wb")
)
