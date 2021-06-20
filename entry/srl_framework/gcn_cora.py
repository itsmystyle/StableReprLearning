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
from tqdm.auto import tqdm
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
# from sklearn.metrics import r2_score

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


# ### utils
# ***

# In[6]:


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
scores = np.array([[[0.0 for _ in range(8)] for _ in range(n)] for _ in range(32)])

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
    model2.cls = model1.cls
    for param in model2.cls.parameters():
        param.requires_grad = False
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

    # 1. calculate features selection using
    # (1 - normalized mse) with embedding_a and embedding_b
    print("extract embedding of first model")
    emb_train_a = extract_embedding(model1, g, features, labels, train_mask)
    emb_test_a = extract_embedding(model1, g, features, labels, test_mask)
    data_a = {
        "train_x": np.array(emb_train_a["embedding"]),
        "train_y": np.array(emb_train_a["target"]),
        "test_x": np.array(emb_test_a["embedding"]),
        "test_y": np.array(emb_test_a["target"]),
    }

    print("extract embedding of second model")
    emb_train_b = extract_embedding(model2, g, features, labels, train_mask)
    emb_test_b = extract_embedding(model2, g, features, labels, test_mask)
    data_b = {
        "train_x": np.array(emb_train_b["embedding"]),
        "train_y": np.array(emb_train_b["target"]),
        "test_x": np.array(emb_test_b["embedding"]),
        "test_y": np.array(emb_test_b["target"]),
    }

    # ----- r2 score
    # r2_score_ls = []
    # for _i in range(data_a['test_x'].shape[1]):
    #     _r2_score = r2_score(data_a['test_x'][:, _i],
    #                          data_b['test_x'][:, _i])
    #     r2_score_ls += [_r2_score]
    # sorted_index = sorted(range(len(r2_score_ls)), key=lambda k: r2_score_ls[k], reverse=True)

    # ----- output weight
    # sorted_index = (
    #         torch.sort(torch.abs(model1.cls.weight).mean(dim=0), descending=True)[1]
    #         .detach()
    #         .cpu()
    #         .numpy()
    #     )

    # ----- permutation importance
    x, y = data_a["train_x"], data_a["train_y"]
    criterion = nn.CrossEntropyLoss()

    loss = {}

    for _feature in tqdm(range(x.shape[-1]), total=x.shape[-1]):
        model1.eval()

        if _feature not in loss:
            loss[_feature] = []

        for _n in range(5):
            batch_copy = x.copy()
            rand_perm = np.random.permutation(x.shape[0])
            batch_copy[:, _feature] = x[rand_perm, _feature]

            train_x, train_y = (
                torch.FloatTensor(batch_copy),
                torch.LongTensor(y),
            )

            train_loader = torch.utils.data.DataLoader(
                Data.TensorDataset(train_x, train_y),
                batch_size=args["batch_size"] * 2,
                num_workers=8,
                shuffle=False,
            )

            _loss = []

            with torch.no_grad():
                correct = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    data, target = Variable(data), Variable(target)
                    output = model1.classify(data)
                    _l = criterion(output, target)
                    _loss.append(_l.item())
            loss[_feature].append(np.mean(_loss))

    loss_ls = [np.mean(v) for k, v in sorted(loss.items())]
    sorted_index = sorted(range(len(loss_ls)), key=lambda k: loss_ls[k], reverse=True)

    # 2. retrain all model only using 16, 32, 64, 128, 192 neurons
    _n = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]
    for counter, num_to_drop in enumerate(_n):
        counter = counter + 2

        neurons = sorted_index[:num_to_drop]

        for task_id in range(1):
            train_x, train_y = (
                data_a["train_x"][:, neurons],
                data_a["train_y"],
            )
            test_x, test_y = (
                data_a["test_x"][:, neurons],
                data_a["test_y"],
            )
            test_x_b, test_y_b = (
                data_b["test_x"][:, neurons],
                data_b["test_y"],
            )
            train_loader = torch.utils.data.DataLoader(
                Data.TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y)),
                batch_size=args["batch_size"],
                num_workers=4,
                shuffle=True,
                **kwargs,
            )
            test_loader = torch.utils.data.DataLoader(
                Data.TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y)),
                batch_size=args["batch_size"] * 2,
                num_workers=4,
                shuffle=False,
                **kwargs,
            )
            test_b_loader = torch.utils.data.DataLoader(
                Data.TensorDataset(torch.FloatTensor(test_x_b), torch.LongTensor(test_y_b)),
                batch_size=args["batch_size"] * 2,
                num_workers=4,
                shuffle=False,
                **kwargs,
            )

            lr = 5e-3
            epochs = 450

            # linear
            model_l = LNet(len(neurons), 7).to(device)
            optimizer = optim.Adam(model_l.parameters(), lr=lr, weight_decay=1e-5)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=10, verbose=True
            )
            patience = es_thd
            best_acc = 0
            best_model = None

            for epoch in range(1, epochs + 1):
                train(epoch, model_l, optimizer)
                correct = test2(model_l, epoch)
                lr_scheduler.step(-1 * correct / len(test_loader.dataset))
                if correct / len(test_loader.dataset) > best_acc:
                    patience = es_thd
                    best_acc = correct / len(test_loader.dataset)
                    best_model = LNet(len(neurons), 7)
                    best_model.load_state_dict(copy.deepcopy(model_l.state_dict()))

            # restore best model
            model_l = LNet(len(neurons), 7)
            model_l.load_state_dict(copy.deepcopy(best_model.state_dict()))
            model_l.to(device)

            correct = test2(model_l, 0)
            scores[counter][i][1] = correct / len(test_loader.dataset)
            correct = testb(model_l)
            scores[counter + 15][i][1] = correct / len(test_b_loader.dataset)
            model_ls.append(model_l)

            # MLP
            lr = 5e-3
            epochs = 450

            model_mlp = MLP(len(neurons), (len(neurons) + 7) // 2, 7).to(device)
            optimizer = optim.Adam(model_mlp.parameters(), lr=lr, weight_decay=1e-5)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=10, verbose=True
            )
            patience = es_thd
            best_acc = 0
            best_model = None

            for epoch in range(1, epochs + 1):
                train(epoch, model_mlp, optimizer)
                correct = test2(model_mlp, epoch)
                lr_scheduler.step(-1 * correct / len(test_loader.dataset))
                if correct / len(test_loader.dataset) > best_acc:
                    patience = es_thd
                    best_acc = correct / len(test_loader.dataset)
                    best_model = MLP(len(neurons), (len(neurons) + 7) // 2, 7)
                    best_model.load_state_dict(copy.deepcopy(model_mlp.state_dict()))

            # restore best model
            model_mlp = MLP(len(neurons), (len(neurons) + 7) // 2, 7)
            model_mlp.load_state_dict(copy.deepcopy(best_model.state_dict()))
            model_mlp.to(device)

            correct = test2(model_mlp, 0)
            scores[counter][i][2] = correct / len(test_loader.dataset)
            correct = testb(model_mlp)
            scores[counter + 15][i][2] = correct / len(test_b_loader.dataset)
            model_mlps.append(model_mlp)

            # linear svm
            cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
            cls_lsvm.fit(train_x, train_y)
            _valid_score = cls_lsvm.score(test_x, test_y)
            scores[counter][i][3] = _valid_score

            _valid_score_b = cls_lsvm.score(test_x_b, test_y_b)
            scores[counter + 15][i][3] = _valid_score_b

            cls_lsvms.append(cls_lsvm)
            print(f"Linear SVM test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

            # svm
            cls_svm = SVC(random_state=args["seed"])
            cls_svm.fit(train_x, train_y)
            _valid_score = cls_svm.score(test_x, test_y)
            scores[counter][i][4] = _valid_score

            _valid_score_b = cls_svm.score(test_x_b, test_y_b)
            scores[counter + 15][i][4] = _valid_score_b

            cls_svms.append(cls_svm)
            print(f"SVM test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

            # decision tree
            cls_dt = DecisionTreeClassifier(random_state=args["seed"])
            cls_dt.fit(train_x, train_y)
            _valid_score = cls_dt.score(test_x, test_y)
            scores[counter][i][5] = _valid_score

            _valid_score_b = cls_dt.score(test_x_b, test_y_b)
            scores[counter + 15][i][5] = _valid_score_b

            cls_dts.append(cls_dt)
            print(f"Decision Tree test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

            # random forest
            cls_rf = RandomForestClassifier(n_estimators=10, random_state=args["seed"], n_jobs=8)
            cls_rf.fit(train_x, train_y)
            _valid_score = cls_rf.score(test_x, test_y)
            scores[counter][i][6] = _valid_score

            _valid_score_b = cls_rf.score(test_x_b, test_y_b)
            scores[counter + 15][i][6] = _valid_score_b

            cls_rfs.append(cls_rf)
            print(f"Random Forest test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

            # lgb
            cls_lgb = LGBMClassifier(random_state=args["seed"], n_jobs=8)
            cls_lgb.fit(
                train_x,
                train_y,
                eval_set=[(test_x, test_y)],
                early_stopping_rounds=100,
                verbose=100,
            )

            _valid_pred = cls_lgb.predict(test_x)
            _valid_score = sum(_valid_pred == test_y) / len(_valid_pred)
            scores[counter][i][7] = _valid_score

            _valid_pred = cls_lgb.predict(test_x_b)
            _valid_score_b = sum(_valid_pred == test_y_b) / len(_valid_pred)
            scores[counter + 15][i][7] = _valid_score_b

            cls_lgbs.append(cls_lgb)
            print(f"LightGBM test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

    t = round(time.time() - t0)
    print("time consumed: {} min {} sec".format(t // 60, t % 60))

    # pickle.dump(scores, open("scores_gcn_cora_d256_fix_important_feature_v14.pkl", "wb"))
    pickle.dump(scores, open("results/scores_gcn_cora_d256_permutation_v21.pkl", "wb"))
