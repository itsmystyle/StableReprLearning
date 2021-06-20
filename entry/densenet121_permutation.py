from __future__ import print_function
import pickle
import sys
import time
import copy
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from thundersvm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as Data
import torchvision.models as models
from torch.autograd import Variable

# from sklearn.metrics import r2_score

args = {}
kwargs = {}
gpu_id = 0
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

torch.manual_seed(args["seed"])

args["base_dir"] = "model/densenet121_adam_permutation_v20_pretrained/fix"
args["model_save_path"] = "model_{}_{}.pkl"
args["train_save_path"] = "train_{}_{}.pkl"
args["test_save_path"] = "test_{}_{}.pkl"

args["perceptron_save_path"] = "perceptron_{}_{}.pkl"
args["mlp_save_path"] = "mlp_{}_{}.pkl"
args["lsvm_save_path"] = "lsvm_{}_{}.pkl"
args["svm_save_path"] = "svm_{}_{}.pkl"
args["dt_save_path"] = "dt_{}_{}.pkl"
args["rf_save_path"] = "rf_{}_{}.pkl"

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


def get_cifar10():
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("data", train=True, download=True, transform=transform_train),
        batch_size=args["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("data", train=False, transform=transform_test),
        batch_size=args["batch_size"] * 2,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


class DenseNet121(nn.Module):
    def __init__(self, emb_size, n_classes=10):
        super(DenseNet121, self).__init__()
        self.dim = emb_size
        self.n_classes = n_classes

        backbone = models.densenet121(pretrained=True)
        self.encoder = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc1 = nn.Linear(1024, self.dim)

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


def test(model, test_loader, task_id):
    model.eval()
    with torch.no_grad():
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


def test2(model, test_loader):
    model.eval()
    with torch.no_grad():
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


scores = np.array(
    [[[[0.0 for _ in range(8)] for _ in range(n)] for _ in range(3)] for _ in range(32)]
)

for i in range(n):
    model_module = DenseNet121
    Path(args["base_dir"] + f"/{i}").mkdir(parents=True, exist_ok=True)

    print("round:", i)
    t0 = time.time()
    # train first model
    print("train first model")
    train_loader, test_loader = get_cifar10()
    model1 = model_module(emb_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args["lr"], weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, factor=0.3, patience=3, verbose=True
    )

    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        train(epoch, model1, optimizer1, train_loader)
        correct = test(model1, test_loader, task_id=0)
        lr_scheduler.step(-1 * correct / len(test_loader.dataset))
        if correct / len(test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(test_loader.dataset)
            best_model = model_module(emb_size)
            best_model.load_state_dict(copy.deepcopy(model1.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model1 = model_module(emb_size).to(device)
    model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model1.to(device)

    # save model
    save_model_path = args["base_dir"] + f"/{i}/" + args["model_save_path"].format(0, 1)
    torch.save(model1.state_dict(), save_model_path)

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

        # save model
        save_model_path = (
            args["base_dir"] + f"/{i}/" + args["perceptron_save_path"].format(task_id, 1)
        )
        torch.save(model_l.state_dict(), save_model_path)

        # MLP
        model_mlp = MLP(emb_size, (emb_size + out_size[task_id]) // 2, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train(epoch, model_mlp, optimizer, train_loader_cls)
            correct = test2(model_mlp, test_loader_cls)

        scores[0][task_id][i][2] = correct / len(test_loader_cls.dataset)
        model_mlps.append(model_mlp)

        # save model
        save_model_path = args["base_dir"] + f"/{i}/" + args["mlp_save_path"].format(task_id, 1)
        torch.save(model_mlp.state_dict(), save_model_path)

        # linear svm
        cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
        cls_lsvm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_lsvm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][3] = _valid_score
        cls_lsvms.append(cls_lsvm)
        print(f"Linear SVM test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["lsvm_save_path"].format(task_id, 1)
        cls_lsvm.save_to_file(save_model_path)

        # svm
        cls_svm = SVC(random_state=args["seed"], gpu_id=int(gpu_id))
        cls_svm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_svm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][4] = _valid_score
        cls_svms.append(cls_svm)
        print(f"SVM test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["svm_save_path"].format(task_id, 1)
        cls_svm.save_to_file(save_model_path)

        # decision tree
        cls_dt = DecisionTreeClassifier(random_state=args["seed"])
        cls_dt.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_dt.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][5] = _valid_score
        cls_dts.append(cls_dt)
        print(f"Decision Tree test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["dt_save_path"].format(task_id, 1)
        with open(save_model_path, "wb") as fout:
            pickle.dump(cls_dt, fout)

        # random forest
        cls_rf = RandomForestClassifier(n_estimators=10, random_state=args["seed"], n_jobs=8)
        cls_rf.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_rf.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][6] = _valid_score
        cls_rfs.append(cls_rf)
        print(f"Random Forest test acc: {_valid_score:.5f}")

        save_model_path = args["base_dir"] + f"/{i}/" + args["rf_save_path"].format(task_id, 1)
        with open(save_model_path, "wb") as fout:
            pickle.dump(cls_rf, fout)

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
    for j in range(1, 2):
        if j == 1:
            jdx = 1
        elif j == 2:
            jdx = 32
        elif j == 3:
            jdx = 63

        print("train model:", jdx)
        train_loader, test_loader = get_cifar10()
        model2 = model_module(emb_size).to(device)
        model2.cls = model1.cls
        model2.cls.weight.requires_grad = False
        model2.cls.bias.requires_grad = False
        optimizer2 = optim.Adam(
            filter(lambda p: p.requires_grad, model2.parameters()), lr=args["lr"], weight_decay=5e-4
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, factor=0.3, patience=3, verbose=True
        )

        patience = es_thd
        best_acc = 0
        best_model = None

        for epoch in range(1, args["epochs"] + 1):
            train(epoch, model2, optimizer2, train_loader)
            correct = test(model2, test_loader, task_id=0)
            lr_scheduler.step(-1 * correct / len(test_loader.dataset))
            if correct / len(test_loader.dataset) > best_acc:
                patience = es_thd
                best_acc = correct / len(test_loader.dataset)
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

        for task_id in range(3):
            scores[jdx][task_id][i][0] = test(model2, test_loader, task_id=task_id) / len(
                test_loader.dataset
            )

        # save model
        save_model_path = args["base_dir"] + f"/{i}/" + args["model_save_path"].format(0, j + 1)
        torch.save(model2.state_dict(), save_model_path)

        # save data for other CLSs
        print("extract embedding")
        emb_test = extract_embedding(model2, test_loader)
        data = {"test_x": np.array(emb_test["embedding"]), "test_y": np.array(emb_test["target"])}

        # save_model_path = args['base_dir'] + f'/{i}/' + args["train_save_path"].format(0, j + 1)
        # with open(save_model_path, 'wb') as fout:
        #     _dict = {"x": data['train_x'], "y": data['train_y']}
        #     pickle.dump(_dict, fout)

        save_model_path = args["base_dir"] + f"/{i}/" + args["test_save_path"].format(0, j + 1)
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
            test_loader_cls = torch.utils.data.DataLoader(
                Data.TensorDataset(test_x, test_y),
                batch_size=args["batch_size"] * 2,
                shuffle=True,
                **kwargs,
            )

            # linear
            correct = test2(model_ls[task_id], test_loader_cls)
            scores[jdx][task_id][i][1] = correct / len(test_loader_cls.dataset)

            # MLP
            correct = test2(model_mlps[task_id], test_loader_cls)
            scores[jdx][task_id][i][2] = correct / len(test_loader_cls.dataset)

            # svm
            _valid_score = cls_lsvms[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[jdx][task_id][i][3] = _valid_score
            print(f"Linear SVM test acc:{_valid_score:.5f}")

            # svm
            _valid_score = cls_svms[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[jdx][task_id][i][4] = _valid_score
            print(f"SVM test acc:{_valid_score:.5f}")

            # decision tree
            _valid_score = cls_dts[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[jdx][task_id][i][5] = _valid_score
            print(f"Decision Tree test acc:{_valid_score:.5f}")

            # random forest
            _valid_score = cls_rfs[task_id].score(data["test_x"], data["test_y"][task_id])
            scores[jdx][task_id][i][6] = _valid_score
            print(f"Random Forest test acc:{_valid_score:.5f}")

            # lgb
            _valid_pred = cls_lgbs[task_id].predict(data["test_x"])
            _valid_score = sum(_valid_pred == data["test_y"][task_id]) / len(_valid_pred)
            scores[jdx][task_id][i][7] = _valid_score
            print(f"LightGBM test acc:{_valid_score:.5f}")

        # 1. calculate features selection using
        # (1 - normalized mse) with embedding_a and embedding_b
        train_loader, test_loader = get_cifar10()

        print("extract embedding of first model")
        emb_train_a = extract_embedding(model1, train_loader)
        emb_test_a = extract_embedding(model1, test_loader)
        data_a = {
            "train_x": np.array(emb_train_a["embedding"]),
            "train_y": np.array(emb_train_a["target"]),
            "test_x": np.array(emb_test_a["embedding"]),
            "test_y": np.array(emb_test_a["target"]),
        }

        print("extract embedding of second model")
        emb_train_b = extract_embedding(model2, train_loader)
        emb_test_b = extract_embedding(model2, test_loader)
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
        #     torch.sort(torch.abs(model1.cls.weight).mean(dim=0), descending=True)[1]
        #     .detach()
        #     .cpu()
        #     .numpy()
        # )

        # ----- permutation importance
        x, y = data_a["train_x"], data_a["train_y"][0]
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
                    batch_size=64 * 2,
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
            counter = jdx + counter + 1

            neurons = sorted_index[:num_to_drop]

            for task_id in range(3):
                print("task:", task_id)
                train_x, train_y = (
                    data_a["train_x"][:, neurons],
                    data_a["train_y"][task_id],
                )
                test_x, test_y = (
                    data_a["test_x"][:, neurons],
                    data_a["test_y"][task_id],
                )
                test_x_b, test_y_b = (
                    data_b["test_x"][:, neurons],
                    data_b["test_y"][task_id],
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

                # linear
                model_l = LNet(len(neurons), out_size[task_id]).to(device)
                optimizer = optim.Adam(model_l.parameters(), lr=lr)
                patience = es_thd
                best_acc = 0
                best_model = None

                for epoch in range(1, epochs + 1):
                    train(epoch, model_l, optimizer, train_loader)
                    correct = test2(model_l, test_loader)
                    if correct / len(test_loader.dataset) > best_acc:
                        patience = es_thd
                        best_acc = correct / len(test_loader.dataset)
                        best_model = LNet(len(neurons), out_size[task_id])
                        best_model.load_state_dict(copy.deepcopy(model_l.state_dict()))
                    else:
                        patience -= 1
                    if patience <= 0:
                        break

                # restore best model
                model_l = LNet(len(neurons), out_size[task_id])
                model_l.load_state_dict(copy.deepcopy(best_model.state_dict()))
                model_l.to(device)

                # save model
                save_model_path = (
                    args["base_dir"]
                    + f"/{i}/"
                    + args["perceptron_save_path"].format(task_id, counter)
                )
                torch.save(model_l.state_dict(), save_model_path)

                correct = test2(model_l, test_loader)
                scores[counter][task_id][i][1] = correct / len(test_loader.dataset)
                correct = testb(model_l)
                scores[counter + 15][task_id][i][1] = correct / len(test_b_loader.dataset)
                model_ls.append(model_l)

                # MLP
                model_mlp = MLP(
                    len(neurons), (len(neurons) + out_size[task_id]) // 2, out_size[task_id]
                ).to(device)
                optimizer = optim.Adam(model_mlp.parameters(), lr=lr)
                patience = es_thd
                best_acc = 0
                best_model = None

                for epoch in range(1, epochs + 1):
                    train(epoch, model_mlp, optimizer, train_loader)
                    correct = test2(model_mlp, test_loader)
                    if correct / len(test_loader.dataset) > best_acc:
                        patience = es_thd
                        best_acc = correct / len(test_loader.dataset)
                        best_model = MLP(
                            len(neurons), (len(neurons) + out_size[task_id]) // 2, out_size[task_id]
                        )
                        best_model.load_state_dict(copy.deepcopy(model_mlp.state_dict()))
                    else:
                        patience -= 1
                    if patience <= 0:
                        break

                # restore best model
                model_mlp = MLP(
                    len(neurons), (len(neurons) + out_size[task_id]) // 2, out_size[task_id]
                )
                model_mlp.load_state_dict(copy.deepcopy(best_model.state_dict()))
                model_mlp.to(device)

                # save model
                save_model_path = (
                    args["base_dir"] + f"/{i}/" + args["mlp_save_path"].format(task_id, counter)
                )
                torch.save(model_mlp.state_dict(), save_model_path)

                correct = test2(model_mlp, test_loader)
                scores[counter][task_id][i][2] = correct / len(test_loader.dataset)
                correct = testb(model_mlp)
                scores[counter + 15][task_id][i][2] = correct / len(test_b_loader.dataset)
                model_mlps.append(model_mlp)

                # linear svm
                cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
                cls_lsvm.fit(train_x, train_y)
                _valid_score = cls_lsvm.score(test_x, test_y)
                scores[counter][task_id][i][3] = _valid_score

                _valid_score_b = cls_lsvm.score(test_x_b, test_y_b)
                scores[counter + 15][task_id][i][3] = _valid_score_b

                cls_lsvms.append(cls_lsvm)
                print(f"Linear SVM test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

                save_model_path = (
                    args["base_dir"] + f"/{i}/" + args["lsvm_save_path"].format(task_id, counter)
                )
                cls_lsvm.save_to_file(save_model_path)

                # svm
                cls_svm = SVC(random_state=args["seed"])
                cls_svm.fit(train_x, train_y)
                _valid_score = cls_svm.score(test_x, test_y)
                scores[counter][task_id][i][4] = _valid_score

                _valid_score_b = cls_svm.score(test_x_b, test_y_b)
                scores[counter + 15][task_id][i][4] = _valid_score_b

                cls_svms.append(cls_svm)
                print(f"SVM test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

                save_model_path = (
                    args["base_dir"] + f"/{i}/" + args["svm_save_path"].format(task_id, counter)
                )
                cls_svm.save_to_file(save_model_path)

                # decision tree
                cls_dt = DecisionTreeClassifier(random_state=args["seed"])
                cls_dt.fit(train_x, train_y)
                _valid_score = cls_dt.score(test_x, test_y)
                scores[counter][task_id][i][5] = _valid_score

                _valid_score_b = cls_dt.score(test_x_b, test_y_b)
                scores[counter + 15][task_id][i][5] = _valid_score_b

                cls_dts.append(cls_dt)
                print(f"Decision Tree test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

                save_model_path = (
                    args["base_dir"] + f"/{i}/" + args["dt_save_path"].format(task_id, counter)
                )
                with open(save_model_path, "wb") as fout:
                    pickle.dump(cls_dt, fout)

                # random forest
                cls_rf = RandomForestClassifier(
                    n_estimators=10, random_state=args["seed"], n_jobs=8
                )
                cls_rf.fit(train_x, train_y)
                _valid_score = cls_rf.score(test_x, test_y)
                scores[counter][task_id][i][6] = _valid_score

                _valid_score_b = cls_rf.score(test_x_b, test_y_b)
                scores[counter + 15][task_id][i][6] = _valid_score_b

                cls_rfs.append(cls_rf)
                print(f"Random Forest test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

                save_model_path = (
                    args["base_dir"] + f"/{i}/" + args["rf_save_path"].format(task_id, counter)
                )
                with open(save_model_path, "wb") as fout:
                    pickle.dump(cls_rf, fout)

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
                scores[counter][task_id][i][7] = _valid_score

                _valid_pred = cls_lgb.predict(test_x_b)
                _valid_score_b = sum(_valid_pred == test_y_b) / len(_valid_pred)
                scores[counter + 15][task_id][i][7] = _valid_score_b

                cls_lgbs.append(cls_lgb)
                print(f"LightGBM test acc: {_valid_score:.5f}, {_valid_score_b:.5f}")

        t = round(time.time() - t0)
        print("time consumed: {} min {} sec".format(t // 60, t % 60))

    file_name = "results/scores_densenet121_v20_important_feature_ADAM_permutation_pretrained.pkl"
    pickle.dump(scores, open(file_name, "wb"))
