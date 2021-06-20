from __future__ import print_function
import pickle
import sys
import time
import copy
import re
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
# from thundersvm import SVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from torch.utils.data import Dataset
from torchnlp.datasets import imdb_dataset
from nltk.tokenize import word_tokenize


args = {}
kwargs = {}
emb_size = 256
es_thd = int(sys.argv[2])
n = int(sys.argv[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

args["batch_size"] = 64
args["epochs"] = 50
args["lr"] = 1e-3
args["seed"] = 5487

torch.manual_seed(args["seed"])


class Embedding:
    """
    Args:
        embedding_path (str): Path where embedding are loaded from (text file).
        words (None or list): If not None, only load embedding of the words in
            the list.
        oov_as_unk (bool): If argument `words` are provided, whether or not
            treat words in `words` but not in embedding file as `<unk>`. If
            true, OOV will be mapped to the index of `<unk>`. Otherwise,
            embedding of those OOV will be randomly initialize and their
            indices will be after non-OOV.
        lower (bool): Whether or not lower the words.
        rand_seed (int): Random seed for embedding initialization.
    """

    def __init__(self, embedding_path, words=None, oov_as_unk=True, lower=True, rand_seed=42):
        self.word_dict = {}
        self.vectors = None
        self.lower = lower
        self.extend(embedding_path, words, oov_as_unk)
        torch.manual_seed(rand_seed)

        if "<pad>" not in self.word_dict:
            self.add("<pad>", torch.zeros(self.get_dim()))

        if "<bos>" not in self.word_dict:
            t_tensor = torch.rand((1, self.get_dim()), dtype=torch.float)
            torch.nn.init.orthogonal_(t_tensor)
            self.add("<bos>", t_tensor)

        if "<eos>" not in self.word_dict:
            t_tensor = torch.rand((1, self.get_dim()), dtype=torch.float)
            torch.nn.init.orthogonal_(t_tensor)
            self.add("<eos>", t_tensor)

        if "<unk>" not in self.word_dict:
            self.add("<unk>")

    def to_index(self, word):
        """
        Args:
            word (str)

        Return:
             index of the word. If the word is not in `words` and not in the
             embedding file, then index of `<unk>` will be returned.
        """
        if self.lower:
            word = word.lower()

        if word not in self.word_dict:
            return self.word_dict["<unk>"]
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.vectors.shape[1]

    def get_vocabulary_size(self):
        return self.vectors.shape[0]

    def add(self, word, vector=None):
        if self.lower:
            word = word.lower()

        if vector is not None:
            vector = vector.view(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.vectors = torch.cat([self.vectors, vector], 0)
        self.word_dict[word] = len(self.word_dict)

    def extend(self, embedding_path, words, oov_as_unk=True):
        self._load_embedding(embedding_path, words)

        if words is not None and not oov_as_unk:
            # initialize word vector for OOV
            for word in words:
                if self.lower:
                    word = word.lower()

                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

            oov_vectors = torch.nn.init.uniform_(
                torch.empty(len(self.word_dict) - self.vectors.shape[0], self.vectors.shape[1])
            )

            self.vectors = torch.cat([self.vectors, oov_vectors], 0)

    def _load_embedding(self, embedding_path, words):
        if words is not None:
            words = set(words)

        vectors = []

        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match("^[0-9]+ [0-9]+$", row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(" ")
                word = cols[0]

                # skip word not in words if words are provided
                if words is not None and word not in words:
                    continue
                elif word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                    vectors.append([float(v) for v in cols[1:]])

        vectors = torch.tensor(vectors)
        if self.vectors is not None:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
        else:
            self.vectors = vectors


class IMDBDataset(Dataset):
    def __init__(self, mode="train_d", num_class=2):

        if mode == "train_d":
            self.data = imdb_dataset(train=True, test=False)

            if os.path.exists("data/imdb_embedding.pkl"):
                with open("data/imdb_embedding.pkl", "rb") as fin:
                    self.embedder = pickle.load(fin)
            else:
                # collect word
                def collect_words(data, n_workers=4):
                    sent_list = []
                    for i in data:
                        sent_list += [i["text"]]

                    chunks = [
                        " ".join(sent_list[i: i + len(sent_list) // n_workers])
                        for i in range(0, len(sent_list), len(sent_list) // n_workers)
                    ]
                    with Pool(n_workers) as pool:
                        chunks = pool.map_async(word_tokenize, chunks)
                        words = set(sum(chunks.get(), []))

                    return words

                words = set()
                words |= collect_words(self.data)
                print(f"{len(words)} words collected...")

                PAD_TOKEN = 0
                UNK_TOKEN = 1
                word_dict = {"<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
                for word in words:
                    word_dict[word] = len(word_dict)

                self.embedder = Embedding("data/glove.6B.300d.txt", words)

                with open("data/imdb_embedding.pkl", "wb") as fout:
                    pickle.dump(self.embedder, fout)

        elif mode == "test" or mode == "valid":
            self.data = imdb_dataset(train=False, test=True)

            with open("data/imdb_embedding.pkl", "rb") as fin:
                self.embedder = pickle.load(fin)
        else:
            raise NotImplementedError

        self.num_class = num_class
        self.max_len = 1100

    @property
    def vocabulary_size(self):
        return self.embedder.get_vocabulary_size()

    @property
    def embedding_dim(self):
        return self.embedder.get_dim()

    def label_to_onehot(self, labels):
        if labels == "pos":
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "X": [self.embedder.to_index(word) for word in word_tokenize(self.data[index]["text"])],
            "y": self.label_to_onehot(self.data[index]["sentiment"]),
        }

    def collate_fn(self, datas):
        # get max length in this batch
        max_len = min(max([len(data["X"]) for data in datas]), self.max_len)
        batch_X = [
            data["X"][:max_len]
            if len(data["X"]) > max_len
            else data["X"] + [self.embedder.to_index("<pad>")] * (max_len - len(data["X"]))
            for data in datas
        ]
        batch_y = [data["y"] for data in datas]

        return torch.LongTensor(batch_X), torch.LongTensor(batch_y)


def get_imdb():
    train_data = IMDBDataset(mode="train_d")
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=8,
        collate_fn=train_data.collate_fn,
    )
    valid_data = IMDBDataset(mode="valid")
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args["batch_size"] * 2,
        shuffle=False,
        num_workers=8,
        collate_fn=valid_data.collate_fn,
    )

    return train_dataloader, valid_dataloader


class GRUNet(nn.Module):
    def __init__(self, embedding, emb_size, n_classes=2):
        super(GRUNet, self).__init__()
        self.hidden_dim = emb_size
        self.output_dim = 256
        self.n_classes = n_classes
        self.embedding = nn.Embedding(
            embedding.get_vocabulary_size(), embedding.get_dim(), _weight=embedding.vectors
        )
        self.encoder = nn.GRU(
            embedding.get_dim(), self.output_dim, bidirectional=True, batch_first=True
        )
        self.bn = nn.BatchNorm1d(self.output_dim * 2)
        self.fc1 = nn.Linear(self.output_dim * 2, self.hidden_dim)
        self.cls = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, return_embs=False):
        z = self.extract_emb(x)
        x = self.classify(z)

        if return_embs:
            return x, z
        return x

    def extract_emb(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = torch.max(x, dim=1)[0]
        x = self.bn(F.relu(x))
        x = self.fc1(x)

        return x

    def classify(self, x):
        return self.cls(x)


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


def train(epoch, model, optimizer):
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


def test(model, task_id):
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            data, target = (
                data.to(device),
                target.to(device),
            )
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            correct += sum((pred == target).tolist())
        print("Accuracy:", correct / len(test_loader.dataset))
    return correct


def test2(model):
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
            ret["target"][0] += [t for t in target.view(-1).tolist()]
    return ret


model_module = GRUNet
scores = np.array(
    [[[[0.0 for _ in range(8)] for _ in range(n)] for _ in range(3)] for _ in range(2)]
)

for i in range(n):
    print("round:", i)
    t0 = time.time()
    # train first model
    print("train first model")
    train_loader, test_loader = get_imdb()

    model1 = model_module(train_loader.dataset.embedder, emb_size).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args["lr"], weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, factor=0.3, patience=3, verbose=True
    )

    patience = es_thd
    best_acc = 0
    best_model = None

    for epoch in range(1, args["epochs"] + 1):
        train(epoch, model1, optimizer1)
        correct = test(model1, task_id=0)
        lr_scheduler.step(-1 * correct / len(test_loader.dataset))
        if correct / len(test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(test_loader.dataset)
            best_model = model_module(train_loader.dataset.embedder, emb_size)
            best_model.load_state_dict(copy.deepcopy(model1.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model1 = model_module(train_loader.dataset.embedder, emb_size)
    model1.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model1.to(device)

    for task_id in range(1):
        scores[0][task_id][i][0] = test(model1, task_id=task_id) / len(test_loader.dataset)
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

    for task_id in range(1):
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

        scores[0][task_id][i][1] = correct / len(test_loader.dataset)
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

        scores[0][task_id][i][2] = correct / len(test_loader.dataset)
        model_mlps.append(model_mlp)

        # linear svm
        cls_lsvm = SVC(kernel="linear", random_state=args["seed"])
        cls_lsvm.fit(data["train_x"], data["train_y"][task_id])
        _valid_score = cls_lsvm.score(data["test_x"], data["test_y"][task_id])
        scores[0][task_id][i][3] = _valid_score
        cls_lsvms.append(cls_lsvm)
        print(f"Linear SVM test acc: {_valid_score:.5f}")

        # svm
        cls_svm = SVC(random_state=args["seed"])
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

    # train second model with first model's CLS
    print("train second model")
    train_loader, test_loader = get_imdb()
    model2 = model_module(train_loader.dataset.embedder, emb_size).to(device)
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
        train(epoch, model2, optimizer2)
        correct = test(model2, task_id=0)
        lr_scheduler.step(-1 * correct / len(test_loader.dataset))
        if correct / len(test_loader.dataset) > best_acc:
            patience = es_thd
            best_acc = correct / len(test_loader.dataset)
            best_model = model_module(train_loader.dataset.embedder, emb_size)
            best_model.load_state_dict(copy.deepcopy(model2.state_dict()))
        else:
            patience -= 1
        if patience <= 0:
            print("Early stopping!")
            break

    # restore best model
    model2 = model_module(train_loader.dataset.embedder, emb_size)
    model2.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model2.to(device)

    for task_id in range(1):
        scores[1][task_id][i][0] = test(model2, task_id=task_id) / len(test_loader.dataset)
    # save data for other CLSs
    print("extract embedding")
    emb_test = extract_embedding(model2, test_loader)
    data = {"test_x": np.array(emb_test["embedding"]), "test_y": np.array(emb_test["target"])}

    # test CLSs
    print("test CLSs")
    for task_id in range(1):
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
        scores[1][task_id][i][1] = correct / len(test_loader.dataset)

        # MLP
        correct = test2(model_mlps[task_id])
        scores[1][task_id][i][2] = correct / len(test_loader.dataset)

        # svm
        _valid_score = cls_lsvms[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][3] = _valid_score
        print(f"Linear SVM test acc:{_valid_score:.5f}")

        # svm
        _valid_score = cls_svms[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][4] = _valid_score
        print(f"SVM test acc:{_valid_score:.5f}")

        # decision tree
        _valid_score = cls_dts[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][5] = _valid_score
        print(f"Decision Tree test acc:{_valid_score:.5f}")

        # random forest
        _valid_score = cls_rfs[task_id].score(data["test_x"], data["test_y"][task_id])
        scores[1][task_id][i][6] = _valid_score
        print(f"Random Forest test acc:{_valid_score:.5f}")

        # lgb
        _valid_pred = cls_lgbs[task_id].predict(data["test_x"])
        _valid_score = sum(_valid_pred == data["test_y"][task_id]) / len(_valid_pred)
        scores[1][task_id][i][7] = _valid_score
        print(f"LightGBM test acc:{_valid_score:.5f}")

    t = round(time.time() - t0)
    print("time consumed: {} min {} sec".format(t // 60, t % 60))

    pickle.dump(scores, open("scores_gru_d256_v23_nofix.pkl", "wb"))
