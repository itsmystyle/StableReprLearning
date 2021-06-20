from __future__ import print_function
import pickle
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from dgl.nn.pytorch import GraphConv
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

args["batch_size"] = 64
args["epochs"] = 800  # 1500
args["lr"] = 8e-3  # 5e-3
args["seed"] = 4896
_wc = 1e-3  # 2e-3

torch.manual_seed(args["seed"])
dgl.random.seed(args["seed"])


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        g, features, labels, mask = (
            g.to(device),
            features.to(device),
            labels.to(device),
            mask.to(device),
        )
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

    return correct.item() * 1.0 / len(labels)


class Net(nn.Module):
    def __init__(self, emb_dim):
        super(Net, self).__init__()
        self.gcn_layer1 = GraphConv(1433, 384)
        self.gcn_layer2 = GraphConv(384, emb_dim)
        self.cls = nn.Linear(emb_dim, 7)

    def forward(self, graph, features):
        h = self.extract_emb(graph, features)
        h = self.classify(h)

        return h

    def extract_emb(self, graph, features):
        h = F.relu(self.gcn_layer1(graph, features))
        h = self.gcn_layer2(graph, h)

        return h

    def classify(self, features):
        return self.cls(features)


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


def traina(epoch, model, optimizer, g, features, labels, train_mask):
    if epoch % 100 == 0:
        print("Epoch:", epoch)
    model.train()

    g, features, labels, train_mask = (
        g.to(device),
        features.to(device),
        labels.to(device),
        train_mask.to(device),
    )
    optimizer.zero_grad()

    logits = model(g, features)
    criterion = nn.CrossEntropyLoss()
    # logp = F.log_softmax(logits, 1)
    # loss = F.nll_loss(logp[train_mask], labels[train_mask])
    loss = criterion(logits[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()


def train(epoch, model, optimizer):
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


def test2(model, epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            correct += sum((pred == target).tolist())
        if epoch % 15 == 0:
            print("Accuracy:", correct / len(test_loader.dataset))
    return correct


def testb(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in test_b_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            correct += sum((pred == target).tolist())
        print("Accuracy:", correct / len(test_loader.dataset))
    return correct


def extract_embedding(model, g, features, labels, mask):

    ret = {"embedding": [], "target": []}
    model.eval()
    with torch.no_grad():
        g, features, labels, mask = (
            g.to(device),
            features.to(device),
            labels.to(device),
            mask.to(device),
        )

        embs = model.extract_emb(g, features)

        emb = embs[mask].cpu().numpy()
        label = labels[mask].cpu().numpy()

        ret["embedding"] = emb
        ret["target"] = label

    return ret


model_module = Net
scores = np.array([[[0.0 for _ in range(8)] for _ in range(n)] for _ in range(2)])

for i in range(n):
    print("round:", i)
    t0 = time.time()
    # train first model
    print("train first model")
    g, features, labels, train_mask, test_mask = load_cora_data()
    model1 = model_module(emb_dim=emb_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args["lr"], weight_decay=_wc)

    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        traina(epoch, model1, optimizer1, g, features, labels, train_mask)
        correct = evaluate(model1, g, features, labels, test_mask)
        if epoch % 100 == 0:
            print(f"{correct} | {best_acc}")
        if correct > best_acc:
            patience = es_thd
            best_acc = correct
            best_model = model_module(emb_dim=emb_size)
            best_model.load_state_dict(copy.deepcopy(model1.state_dict()))

    # restore best model
    model1 = model_module(emb_dim=emb_size)
    model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model1.to(device)

    scores[0][i][0] = evaluate(model1, g, features, labels, test_mask)

    # save data for other CLSs
    print("extract embedding")

    emb_train = extract_embedding(model1, g, features, labels, train_mask)
    emb_test = extract_embedding(model1, g, features, labels, test_mask)
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
    lr = 5e-3
    epochs = 250

    for task_id in range(1):
        # training & testing data
        train_x, train_y = (
            torch.FloatTensor(data["train_x"]),
            torch.LongTensor(data["train_y"]),
        )
        test_x, test_y = (
            torch.FloatTensor(data["test_x"]),
            torch.LongTensor(data["test_y"]),
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

        # linear
        model_l = LNet(emb_size, 7).to(device)
        optimizer = optim.Adam(model_l.parameters(), lr=lr, weight_decay=5e-5)
        patience = es_thd
        best_acc = 0
        best_model = None

        for epoch in range(1, epochs + 1):
            train(epoch, model_l, optimizer)
            correct = test2(model_l, epoch)
            if correct / len(test_loader.dataset) > best_acc:
                patience = es_thd
                best_acc = correct / len(test_loader.dataset)
                best_model = LNet(emb_size, 7)
                best_model.load_state_dict(copy.deepcopy(model_l.state_dict()))

        # restore best model
        model_l = LNet(emb_size, 7)
        model_l.load_state_dict(copy.deepcopy(best_model.state_dict()))
        model_l.to(device)

        scores[0][i][1] = correct / len(test_loader.dataset)
        model_ls.append(model_l)

        # MLP
        model_mlp = MLP(emb_size, (emb_size + 7) // 2, 7).to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=lr, weight_decay=5e-5)
        patience = es_thd
        best_acc = 0
        best_model = None

        for epoch in range(1, epochs + 1):
            train(epoch, model_mlp, optimizer)
            correct = test2(model_mlp, epoch)
            if correct / len(test_loader.dataset) > best_acc:
                patience = es_thd
                best_acc = correct / len(test_loader.dataset)
                best_model = MLP(emb_size, (emb_size + 7) // 2, 7)
                best_model.load_state_dict(copy.deepcopy(model_mlp.state_dict()))

        # restore best model
        model_mlp = MLP(emb_size, (emb_size + 7) // 2, 7)
        model_mlp.load_state_dict(copy.deepcopy(best_model.state_dict()))
        model_mlp.to(device)

        scores[0][i][2] = correct / len(test_loader.dataset)
        model_mlps.append(model_mlp)

        # linear svm
        cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
        cls_lsvm.fit(data["train_x"], data["train_y"])
        _valid_score = cls_lsvm.score(data["test_x"], data["test_y"])
        scores[0][i][3] = _valid_score
        cls_lsvms.append(cls_lsvm)
        print(f"Linear SVM test acc: {_valid_score:.5f}")

        # svm
        cls_svm = SVC(random_state=args["seed"])
        cls_svm.fit(data["train_x"], data["train_y"])
        _valid_score = cls_svm.score(data["test_x"], data["test_y"])
        scores[0][i][4] = _valid_score
        cls_svms.append(cls_svm)
        print(f"SVM test acc: {_valid_score:.5f}")

        # decision tree
        cls_dt = DecisionTreeClassifier(random_state=args["seed"])
        cls_dt.fit(data["train_x"], data["train_y"])
        _valid_score = cls_dt.score(data["test_x"], data["test_y"])
        scores[0][i][5] = _valid_score
        cls_dts.append(cls_dt)
        print(f"Decision Tree test acc: {_valid_score:.5f}")

        # random forest
        cls_rf = RandomForestClassifier(n_estimators=10, random_state=args["seed"], n_jobs=8)
        cls_rf.fit(data["train_x"], data["train_y"])
        _valid_score = cls_rf.score(data["test_x"], data["test_y"])
        scores[0][i][6] = _valid_score
        cls_rfs.append(cls_rf)
        print(f"Random Forest test acc: {_valid_score:.5f}")

        # lgb
        cls_lgb = LGBMClassifier(random_state=args["seed"], n_jobs=8)
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
        cls_lgbs.append(cls_lgb)
        print(f"LightGBM test acc: {_valid_score:.5f}")

    # train second model with first model's CLS
    print("train second model")
    # g, features, labels, train_mask, test_mask = load_cora_data()
    model2 = model_module(emb_dim=emb_size).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=args["lr"], weight_decay=_wc)

    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):

        traina(epoch, model2, optimizer2, g, features, labels, train_mask)
        correct = evaluate(model2, g, features, labels, test_mask)
        if epoch % 100 == 0:
            print(f"{correct} | {best_acc}")
        if correct > best_acc:
            patience = es_thd
            best_acc = correct
            best_model = model_module(emb_dim=emb_size)
            best_model.load_state_dict(copy.deepcopy(model2.state_dict()))

    # restore best model
    model2 = model_module(emb_dim=emb_size)
    model2.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model2.to(device)

    scores[1][i][0] = evaluate(model1, g, features, labels, test_mask)

    # save data for other CLSs
    print("extract embedding")
    emb_test = extract_embedding(model2, g, features, labels, test_mask)
    data = {
        "test_x": np.array(emb_test["embedding"]),
        "test_y": np.array(emb_test["target"]),
    }

    # test CLSs
    print("test CLSs")
    for task_id in range(1):
        # embedding
        test_x, test_y = (
            torch.FloatTensor(data["test_x"]),
            torch.LongTensor(data["test_y"]),
        )
        test_loader = torch.utils.data.DataLoader(
            Data.TensorDataset(test_x, test_y),
            batch_size=args["batch_size"] // 2,
            shuffle=False,
            **kwargs,
        )

        # linear
        correct = test2(model_ls[-1], 0)
        scores[1][i][1] = correct / len(test_loader.dataset)

        # MLP
        correct = test2(model_mlps[-1], 0)
        scores[1][i][2] = correct / len(test_loader.dataset)

        # svm
        _valid_score = cls_lsvms[-1].score(data["test_x"], data["test_y"])
        scores[1][i][3] = _valid_score
        print(f"Linear SVM test acc:{_valid_score:.5f}")

        # svm
        _valid_score = cls_svms[-1].score(data["test_x"], data["test_y"])
        scores[1][i][4] = _valid_score
        print(f"SVM test acc:{_valid_score:.5f}")

        # decision tree
        _valid_score = cls_dts[-1].score(data["test_x"], data["test_y"])
        scores[1][i][5] = _valid_score
        print(f"Decision Tree test acc:{_valid_score:.5f}")

        # random forest
        _valid_score = cls_rfs[-1].score(data["test_x"], data["test_y"])
        scores[1][i][6] = _valid_score
        print(f"Random Forest test acc:{_valid_score:.5f}")

        # lgb
        _valid_pred = cls_lgbs[-1].predict(data["test_x"])
        _valid_score = sum(_valid_pred == data["test_y"]) / len(_valid_pred)
        scores[1][i][7] = _valid_score
        print(f"LightGBM test acc:{_valid_score:.5f}")

    t = round(time.time() - t0)
    print("time consumed: {} min {} sec".format(t // 60, t % 60))

    pickle.dump(scores, open("results/scores_gcn_cora_d256_nofix_v21.pkl", "wb"))
