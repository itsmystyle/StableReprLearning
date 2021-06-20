#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import pickle, sys, time, copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.models as models
from tqdm.auto import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

args = {}
kwargs = {}
emb_size = 256
es_thd = int(sys.argv[2])
n = int(sys.argv[1])
trans_dicts = [
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0},
    {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 1},
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

args["batch_size"] = 128
args["epochs"] = 200
args["lr"] = 1e-3
args["seed"] = 4896
args["save_all"] = False
args["base_dir"] = "model/vgg16/class_incremental/fix/"
args["model_save_path"] = args["base_dir"] + "model_{}_{}.pkl"
args["train_save_path"] = args["base_dir"] + "train_{}_{}.pkl"
args["test_save_path"] = args["base_dir"] + "test_{}_{}.pkl"

args["perceptron_save_path"] = args["base_dir"] + "perceptron_{}_{}.pkl"
args["mlp_save_path"] = args["base_dir"] + "mlp_{}_{}.pkl"
args["lsvm_save_path"] = args["base_dir"] + "lsvm_{}_{}.pkl"
args["svm_save_path"] = args["base_dir"] + "svm_{}_{}.pkl"
args["dt_save_path"] = args["base_dir"] + "dt_{}_{}.pkl"
args["rf_save_path"] = args["base_dir"] + "rf_{}_{}.pkl"

random.seed(args["seed"])
torch.manual_seed(args["seed"])

transform_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CIFAR10_8_1(Dataset):
    def __init__(self, train):
        self.train = train
        data = datasets.CIFAR10("data", train=train, transform=None)
        self.dataset = data.data
        self.labels = data.targets

        _dataset = []
        _labels = []
        # self.label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
        self.label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        for d, l in zip(self.dataset, self.labels):
            if l not in [8, 9]:
                _dataset.append(d)
                _labels.append(l)

        self.dataset = _dataset
        self.labels = _labels

        if train:
            X_train, _, y_train, _ = train_test_split(
                self.dataset, self.labels, test_size=0.05, random_state=42, stratify=self.labels
            )
            self.dataset = X_train
            self.labels = y_train

        if self.train:
            self.transform = transform_train
        else:
            self.transform = transform_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            self.transform(self.dataset[index]),
            torch.tensor(self.label_map[self.labels[index]], dtype=torch.long),
        )


class CIFAR10_8_2(Dataset):
    def __init__(self, train):
        self.train = train
        data = datasets.CIFAR10("data", train=train, transform=None)
        self.dataset = data.data
        self.labels = data.targets

        _dataset = []
        _labels = []
        # self.label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
        self.label_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7}
        for d, l in zip(self.dataset, self.labels):
            if l not in [0, 1]:
                _dataset.append(d)
                _labels.append(l)

        self.dataset = _dataset
        self.labels = _labels

        if self.train:
            self.transform = transform_train
        else:
            self.transform = transform_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            self.transform(self.dataset[index]),
            torch.tensor(self.label_map[self.labels[index]], dtype=torch.long),
        )


class CIFAR10_8_3(Dataset):
    def __init__(self, train):
        self.train = train
        data = datasets.CIFAR10("data", train=train, transform=None)
        self.dataset = data.data
        self.labels = data.targets

        _dataset = []
        _labels = []
        self.label_map = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        for d, l in zip(self.dataset, self.labels):
            if l not in [0, 1, 8, 9]:
                _dataset.append(d)
                _labels.append(l)

        self.dataset = _dataset
        self.labels = _labels

        if self.train:
            self.transform = transform_train
        else:
            self.transform = transform_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            self.transform(self.dataset[index]),
            torch.tensor(self.label_map[self.labels[index]], dtype=torch.long),
        )


def imsave(img, path):
    torchvision.utils.save_image(img.cpu(), path)


def get_cifar10():
    upstream_train_loader_1 = torch.utils.data.DataLoader(
        CIFAR10_8_1(train=True),
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=True,
        **kwargs,
    )
    upstream_test_loader_1 = torch.utils.data.DataLoader(
        CIFAR10_8_1(train=False),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )

    upstream_train_loader_2 = torch.utils.data.DataLoader(
        CIFAR10_8_2(train=True),
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=True,
        **kwargs,
    )
    upstream_test_loader_2 = torch.utils.data.DataLoader(
        CIFAR10_8_2(train=False),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )

    downstream_train_loader_1 = torch.utils.data.DataLoader(
        CIFAR10_8_1(train=True),
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=True,
        **kwargs,
    )
    downstream_test_loader_1 = torch.utils.data.DataLoader(
        CIFAR10_8_1(train=False),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )

    downstream_test_loader_2 = torch.utils.data.DataLoader(
        CIFAR10_8_3(train=False),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )
    return (
        upstream_train_loader_1,
        upstream_test_loader_1,
        upstream_train_loader_2,
        upstream_test_loader_2,
        downstream_train_loader_1,
        downstream_test_loader_1,
        downstream_test_loader_2,
    )


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
        self.cls = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.cls(self.fc(x))


def train(epoch, model, optimizer, train_loader, fix_special_weight=False, verbose=True):
    print("Epoch:", epoch)
    model.train()
    criterion = nn.CrossEntropyLoss()
    if verbose:
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if fix_special_weight:
                model.cls.weight.grad[:-2] = 0.0
                model.cls.bias.grad[:-2] = 0.0
            optimizer.step()
    else:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if fix_special_weight:
                model.cls.weight.grad[:-2] = 0.0
                model.cls.bias.grad[:-2] = 0.0
            optimizer.step()


def test(model, task_id, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in tqdm(test_loader):
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


def test2(model, test_loader):
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
    ret = {"embedding": [], "target": []}
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data = Variable(data.to(device))
            output = model.extract_emb(data)
            ret["embedding"] += output.tolist()
            ret["target"] += target.view(-1).tolist()
    return ret


model_module = VGG16
scores = np.array([[[0.0 for _ in range(8)] for _ in range(n)] for _ in range(4)])

for i in range(n):
    print("round:", i)
    t0 = time.time()
    # train first model
    print("train first model")
    (
        upstream_train_loader,
        upstream_test_loader,
        _,
        _,
        downstream_train_loader,
        downstream_test_loader,
        downstream_test_loader_overlap,
    ) = get_cifar10()
    n_classes = 8
    model1 = model_module(emb_size, n_classes=n_classes).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args["lr"])  # weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, factor=0.3, patience=3, verbose=True
    )

    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        train(epoch, model1, optimizer1, upstream_train_loader)
        correct = test2(model1, upstream_test_loader)
        lr_scheduler.step(-1 * correct / len(upstream_test_loader.dataset))
        if correct / len(upstream_test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(upstream_test_loader.dataset)
            best_model = model_module(emb_size, n_classes=n_classes)
            best_model.load_state_dict(copy.deepcopy(model1.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model1 = model_module(emb_size, n_classes=n_classes)
    model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model1.to(device)

    print(model1.cls)

    if args["save_all"]:
        torch.save(model1.state_dict(), args["model_save_path"].format(i, 1))

    scores[0][i][0] = test2(model1, upstream_test_loader) / len(upstream_test_loader.dataset)
    scores[2][i][0] = test2(model1, downstream_test_loader_overlap) / len(
        downstream_test_loader_overlap.dataset
    )

    # save data for other CLSs
    print("extract embedding")
    emb_train = extract_embedding(model1, downstream_train_loader)
    emb_test = extract_embedding(model1, downstream_test_loader)
    emb_test_overlap = extract_embedding(model1, downstream_test_loader_overlap)
    data = {
        "train_x": np.array(emb_train["embedding"]),
        "train_y": np.array(emb_train["target"]),
        "test_x": np.array(emb_test["embedding"]),
        "test_y": np.array(emb_test["target"]),
        "test_x_overlap": np.array(emb_test_overlap["embedding"]),
        "test_y_overlap": np.array(emb_test_overlap["target"]),
    }

    if args["save_all"]:
        with open(args["train_save_path"].format(i, 1), "wb") as fout:
            _dict = {"x": data["train_x"], "y": data["train_y"]}
            pickle.dump(_dict, fout)

        with open(args["test_save_path"].format(i, 1), "wb") as fout:
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
    lr = 1e-3
    epochs = 50

    # training & testing data
    train_x, train_y = (
        torch.FloatTensor(data["train_x"]),
        torch.LongTensor(data["train_y"]),
    )
    test_x, test_y = (
        torch.FloatTensor(data["test_x"]),
        torch.LongTensor(data["test_y"]),
    )
    test_x_overlap, test_y_overlap = (
        torch.FloatTensor(data["test_x_overlap"]),
        torch.LongTensor(data["test_y_overlap"]),
    )
    train_loader = torch.utils.data.DataLoader(
        Data.TensorDataset(train_x, train_y),
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        Data.TensorDataset(test_x, test_y),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )
    test_overlap_loader = torch.utils.data.DataLoader(
        Data.TensorDataset(test_x_overlap, test_y_overlap),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )

    # linear
    model_l = LNet(emb_size, n_classes).to(device)
    optimizer = optim.Adam(model_l.parameters(), lr=lr)
    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, epochs + 1):
        train(epoch, model_l, optimizer, train_loader, verbose=False)
        correct = test2(model_l, test_loader)
        if correct / len(test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(test_loader.dataset)
            best_model = LNet(emb_size, n_classes)
            best_model.load_state_dict(copy.deepcopy(model_l.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            break

    # restore best model
    model_l = LNet(emb_size, n_classes)
    model_l.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model_l.to(device)

    scores[0][i][1] = correct / len(test_loader.dataset)
    correct = test2(model_l, test_overlap_loader)
    scores[2][i][1] = correct / len(test_overlap_loader.dataset)
    model_ls.append(model_l)

    print(model_ls[0].cls)

    if args["save_all"]:
        torch.save(model_l.state_dict(), args["perceptron_save_path"].format(i, 1))

    # MLP
    model_mlp = MLP(emb_size, (emb_size + n_classes) // 2, n_classes).to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=lr)
    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, epochs + 1):
        train(epoch, model_mlp, optimizer, train_loader, verbose=False)
        correct = test2(model_mlp, test_loader)
        if correct / len(test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(test_loader.dataset)
            best_model = MLP(emb_size, (emb_size + n_classes) // 2, n_classes)
            best_model.load_state_dict(copy.deepcopy(model_mlp.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            break

    # restore best model
    model_mlp = MLP(emb_size, (emb_size + n_classes) // 2, n_classes)
    model_mlp.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model_mlp.to(device)

    scores[0][i][2] = correct / len(test_loader.dataset)
    correct = test2(model_mlp, test_overlap_loader)
    scores[2][i][2] = correct / len(test_overlap_loader.dataset)
    model_mlps.append(model_mlp)

    print(model_mlps[0].cls)

    if args["save_all"]:
        torch.save(model_mlp.state_dict(), args["mlp_save_path"].format(i, 1))

    # linear svm
    cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
    cls_lsvm.fit(data["train_x"], data["train_y"])
    _valid_score = cls_lsvm.score(data["test_x"], data["test_y"])
    scores[0][i][3] = _valid_score
    print(f"Linear SVM test acc: {_valid_score:.5f}")
    _valid_score = cls_lsvm.score(data["test_x_overlap"], data["test_y_overlap"])
    scores[2][i][3] = _valid_score
    cls_lsvms.append(cls_lsvm)
    print(f"Linear SVM test acc: {_valid_score:.5f}")

    if args["save_all"]:
        with open(args["lsvm_save_path"].format(i, 1), "wb") as fout:
            pickle.dump(cls_lsvm, fout)

    # svm
    cls_svm = SVC(random_state=args["seed"])
    cls_svm.fit(data["train_x"], data["train_y"])
    _valid_score = cls_svm.score(data["test_x"], data["test_y"])
    scores[0][i][4] = _valid_score
    print(f"SVM test acc: {_valid_score:.5f}")
    _valid_score = cls_svm.score(data["test_x_overlap"], data["test_y_overlap"])
    scores[2][i][4] = _valid_score
    cls_svms.append(cls_svm)
    print(f"SVM test acc: {_valid_score:.5f}")

    if args["save_all"]:
        with open(args["svm_save_path"].format(i, 1), "wb") as fout:
            pickle.dump(cls_svm, fout)

    # decision tree
    cls_dt = DecisionTreeClassifier(random_state=args["seed"])
    cls_dt.fit(data["train_x"], data["train_y"])
    _valid_score = cls_dt.score(data["test_x"], data["test_y"])
    scores[0][i][5] = _valid_score
    print(f"Decision Tree test acc: {_valid_score:.5f}")
    _valid_score = cls_dt.score(data["test_x_overlap"], data["test_y_overlap"])
    scores[2][i][5] = _valid_score
    cls_dts.append(cls_dt)
    print(f"Decision Tree test acc: {_valid_score:.5f}")

    if args["save_all"]:
        with open(args["dt_save_path"].format(i, 1), "wb") as fout:
            pickle.dump(cls_dt, fout)

    # random forest
    cls_rf = RandomForestClassifier(n_estimators=10, random_state=args["seed"], n_jobs=16)
    cls_rf.fit(data["train_x"], data["train_y"])
    _valid_score = cls_rf.score(data["test_x"], data["test_y"])
    scores[0][i][6] = _valid_score
    print(f"Random Forest test acc: {_valid_score:.5f}")
    _valid_score = cls_rf.score(data["test_x_overlap"], data["test_y_overlap"])
    scores[2][i][6] = _valid_score
    cls_rfs.append(cls_rf)
    print(f"Random Forest test acc: {_valid_score:.5f}")

    if args["save_all"]:
        with open(args["rf_save_path"].format(i, 1), "wb") as fout:
            pickle.dump(cls_rf, fout)

    # lgb
    cls_lgb = LGBMClassifier(random_state=args["seed"], n_jobs=16)
    cls_lgb.fit(
        data["train_x"],
        data["train_y"],
        eval_set=[(data["test_x"], data["test_y"])],
        early_stopping_rounds=100,
        verbose=100,
    )
    _valid_pred = cls_lgb.predict(data["test_x"])
    _valid_score = sum(_valid_pred == data["test_y"]) / len(_valid_pred)
    scores[0][i][7] = _valid_score
    print(f"LightGBM test acc: {_valid_score:.5f}")
    _valid_pred = cls_lgb.predict(data["test_x_overlap"])
    _valid_score = sum(_valid_pred == data["test_y_overlap"]) / len(_valid_pred)
    scores[2][i][7] = _valid_score
    cls_lgbs.append(cls_lgb)
    print(f"LightGBM test acc: {_valid_score:.5f}")

    # train second model with first model's CLS
    print("train second model")
    (
        _,
        _,
        upstream_train_loader,
        upstream_test_loader,
        downstream_train_loader,
        downstream_test_loader,
        downstream_test_loader_overlap,
    ) = get_cifar10()
    n_classes = 8
    model2 = model_module(emb_size, n_classes=n_classes).to(device)
    model2.cls.weight[:-2].data.copy_(model1.cls.weight[2:].data)
    model2.cls.bias[:-2].data.copy_(model1.cls.bias[2:].data)
    model2.to(device)
    fix_special_weight = True
    optimizer2 = optim.Adam(
        filter(lambda p: p.requires_grad, model2.parameters()), lr=args["lr"],  # weight_decay=5e-4
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, factor=0.3, patience=3, verbose=True
    )

    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        train(
            epoch, model2, optimizer2, upstream_train_loader, fix_special_weight=fix_special_weight
        )
        correct = test2(model2, upstream_test_loader)
        lr_scheduler.step(-1 * correct / len(upstream_test_loader.dataset))
        if correct / len(upstream_test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(upstream_test_loader.dataset)
            best_model = model_module(emb_size, n_classes=n_classes)
            best_model.load_state_dict(copy.deepcopy(model2.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model2 = model_module(emb_size, n_classes=n_classes)
    model2.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model2.to(device)

    print(model2.cls)

    scores[1][i][0] = test2(model2, upstream_test_loader) / len(upstream_test_loader.dataset)
    scores[3][i][0] = test2(model2, downstream_test_loader_overlap) / len(
        downstream_test_loader_overlap.dataset
    )

    if args["save_all"]:
        torch.save(model2.state_dict(), args["model_save_path"].format(i, 2))

    # save data for other CLSs
    print("extract embedding")
    emb_train = extract_embedding(model2, downstream_train_loader)
    emb_test = extract_embedding(model2, downstream_test_loader)
    emb_test_overlap = extract_embedding(model2, downstream_test_loader_overlap)
    data = {
        "train_x": np.array(emb_train["embedding"]),
        "train_y": np.array(emb_train["target"]),
        "test_x": np.array(emb_test["embedding"]),
        "test_y": np.array(emb_test["target"]),
        "test_x_overlap": np.array(emb_test_overlap["embedding"]),
        "test_y_overlap": np.array(emb_test_overlap["target"]),
    }

    if args["save_all"]:
        with open(args["train_save_path"].format(i, 2), "wb") as fout:
            _dict = {"x": data["train_x"], "y": data["train_y"]}
            pickle.dump(_dict, fout)

        with open(args["test_save_path"].format(i, 2), "wb") as fout:
            _dict = {"x": data["test_x"], "y": data["test_y"]}
            pickle.dump(_dict, fout)

    # test CLSs
    print("test CLSs")
    # embedding
    test_x, test_y = (
        torch.FloatTensor(data["test_x"]),
        torch.LongTensor(data["test_y"]),
    )
    test_x_overlap, test_y_overlap = (
        torch.FloatTensor(data["test_x_overlap"]),
        torch.LongTensor(data["test_y_overlap"]),
    )
    test_loader = torch.utils.data.DataLoader(
        Data.TensorDataset(test_x, test_y),
        batch_size=args["batch_size"] * 2,
        shuffle=False,
        **kwargs,
    )
    test_overlap_loader = torch.utils.data.DataLoader(
        Data.TensorDataset(test_x_overlap, test_y_overlap),
        batch_size=args["batch_size"] * 2,
        num_workers=4,
        shuffle=False,
        **kwargs,
    )

    task_id = 0

    # linear
    correct = test2(model_ls[task_id], test_loader)
    scores[1][i][1] = correct / len(test_loader.dataset)
    correct = test2(model_ls[task_id], test_overlap_loader)
    scores[3][i][1] = correct / len(test_overlap_loader.dataset)

    # MLP
    correct = test2(model_mlps[task_id], test_loader)
    scores[1][i][2] = correct / len(test_loader.dataset)
    correct = test2(model_mlps[task_id], test_overlap_loader)
    scores[3][i][2] = correct / len(test_overlap_loader.dataset)

    # lsvm
    _valid_score = cls_lsvms[task_id].score(data["test_x"], data["test_y"])
    scores[1][i][3] = _valid_score
    print(f"Linear SVM test acc:{_valid_score:.5f}")
    _valid_score = cls_lsvms[task_id].score(data["test_x_overlap"], data["test_y_overlap"])
    scores[3][i][3] = _valid_score
    print(f"Linear SVM test acc:{_valid_score:.5f}")

    # svm
    _valid_score = cls_svms[task_id].score(data["test_x"], data["test_y"])
    scores[1][i][4] = _valid_score
    print(f"SVM test acc:{_valid_score:.5f}")
    _valid_score = cls_svms[task_id].score(data["test_x_overlap"], data["test_y_overlap"])
    scores[3][i][4] = _valid_score
    print(f"SVM test acc:{_valid_score:.5f}")

    # decision tree
    _valid_score = cls_dts[task_id].score(data["test_x"], data["test_y"])
    scores[1][i][5] = _valid_score
    print(f"Decision Tree test acc:{_valid_score:.5f}")
    _valid_score = cls_dts[task_id].score(data["test_x_overlap"], data["test_y_overlap"])
    scores[3][i][5] = _valid_score
    print(f"Decision Tree test acc:{_valid_score:.5f}")

    # random forest
    _valid_score = cls_rfs[task_id].score(data["test_x"], data["test_y"])
    scores[1][i][6] = _valid_score
    print(f"Random Forest test acc:{_valid_score:.5f}")
    _valid_score = cls_rfs[task_id].score(data["test_x_overlap"], data["test_y_overlap"])
    scores[3][i][6] = _valid_score
    print(f"Random Forest test acc:{_valid_score:.5f}")

    # lgb
    _valid_pred = cls_lgbs[task_id].predict(data["test_x"])
    _valid_score = sum(_valid_pred == data["test_y"]) / len(_valid_pred)
    scores[1][i][7] = _valid_score
    print(f"LightGBM test acc:{_valid_score:.5f}")
    _valid_pred = cls_lgbs[task_id].predict(data["test_x_overlap"])
    _valid_score = sum(_valid_pred == data["test_y_overlap"]) / len(_valid_pred)
    scores[3][i][7] = _valid_score
    print(f"LightGBM test acc:{_valid_score:.5f}")

    t = round(time.time() - t0)
    print("time consumed: {} min {} sec".format(t // 60, t % 60))

pickle.dump(scores, open("scores_vgg16_class_incremental_88_real_case_v2.pkl", "wb"))
