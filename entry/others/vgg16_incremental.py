#!/usr/bin/env python
# coding: utf-8
# python3 vgg16_incremental.py 16 8400 1 2
# 8400, 7750

# ### imports
# ***

# In[1]:


from __future__ import print_function
import pickle, sys, time
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
from torch.autograd import Variable


# ### parameters
# ***

# In[2]:


args = {}
kwargs = {}
gpu_id = sys.argv[4]
emb_size = 256
es_thd = int(sys.argv[2])
n = int(sys.argv[1])
overlap = bool(int(sys.argv[3]))
n_model = 3 if overlap else 4
trans_dicts = [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
               {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0},
               {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 1}]
device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else 'cpu')
print('using', device)

args['batch_size'] = 128
args['epochs'] = 100
args['lr'] = 0.05
args['seed'] = 4896

torch.manual_seed(args['seed'])


# ### load data for first model
# ***

# In[3]:


transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


# In[4]:


def get_cifar10():
    train_all = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    train_x_all = torch.FloatTensor([train_all.__getitem__(i)[0].tolist() for i in tqdm(range(len(train_all)))])
    train_y_all = torch.LongTensor([train_all.__getitem__(i)[1] for i in tqdm(range(len(train_all)))])
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data',
                                                                train=False,
                                                                transform=transform_test),
                                               batch_size=args['batch_size']*2,
                                               shuffle=True, **kwargs)
    return train_x_all, train_y_all, test_loader

# ### model
# ***

# In[5]:


cfg = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],}

class VGG(nn.Module):
    def __init__(self, emb_size, vgg_name='VGG16'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512, emb_size)
        self.cls = nn.Linear(emb_size, 10)

    def forward(self, x):
        out = self.extract_emb(x)
        out = self.classify(out)
        return out

    def extract_emb(self, x):
        x = self.features(x)
        x = self.fc1(x.view(x.size(0), -1))

        return x

    def classify(self, x):
        x = self.cls(x)

        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x)]
                if i != len(cfg)-2: layers += [nn.ReLU(inplace=True)]
#                 layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
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
    print('Epoch:', epoch)
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

def test(model, test_loader, task_id):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), torch.LongTensor([trans_dicts[task_id][t.item()] for t in target]).to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].view(-1)
            pred = torch.LongTensor([trans_dicts[task_id][p.item()] for p in pred]).to(device)
            correct += sum((pred == target).tolist())
        print('Accuracy:', correct/len(test_loader.dataset))
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
        print('Accuracy:', correct/len(test_loader.dataset))
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


scores = np.array([[[[0.0 for _ in range(7)] for _ in range(n)] for _ in range(3)] for _ in range(n_model)])
train_x_all, train_y_all, test_loader = get_cifar10()
step_size = train_x_all.size(0)//(n_model+1)*2 if overlap else train_x_all.size(0)//n_model
print('step size:', step_size)

for i in range(n):
    print('round:', i)
    t0 = time.time()
    # train first model
    print('train first model')
    train_subset = Data.TensorDataset(train_x_all[:step_size], train_y_all[:step_size])
    train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=args['batch_size'],
                                               shuffle=True, **kwargs)
    model1 = VGG(emb_size).to(device)
    optimizer1 = optim.SGD(model1.parameters(),
                           lr=args['lr'],
                           momentum=0.9,
                           weight_decay=7e-4)

    for epoch in range(1, args['epochs']+1):
        lr = args['lr'] * (0.5**(epoch//30))
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        train(epoch, model1, optimizer1, train_loader)
        correct = test(model1, test_loader, task_id=0)
        if correct >= es_thd:
            print('Early stopping!')
            break

    for task_id in range(3):
        scores[0][task_id][i][0] = test(model1, test_loader, task_id=task_id)/len(test_loader.dataset)
    # save data for other CLSs
    print('extract embedding')
    emb_train = extract_embedding(model1, train_loader)
    emb_test = extract_embedding(model1, test_loader)
    data = {'train_x': np.array(emb_train['embedding']),
            'train_y': np.array(emb_train['target']),
            'test_x': np.array(emb_test['embedding']),
            'test_y': np.array(emb_test['target'])}

    # train CLSs
    print('train CLSs')
    model_ls = []
    model_mlps = []
    cls_svms = []
    cls_dts = []
    cls_rfs = []
    cls_lgbs = []
    out_size = {0: 10, 1: 2, 2: 3}
    lr = 1e-3
    epochs = 10
    
    for task_id in range(3):
        print('task:', task_id)
        # training & testing data
        train_x, train_y = torch.FloatTensor(data['train_x']), torch.LongTensor(data['train_y'][task_id])
        test_x, test_y = torch.FloatTensor(data['test_x']), torch.LongTensor(data['test_y'][task_id])
        train_loader_cls = torch.utils.data.DataLoader(Data.TensorDataset(train_x, train_y),
                                                       batch_size=args['batch_size'],
                                                       shuffle=True, **kwargs)
        test_loader_cls = torch.utils.data.DataLoader(Data.TensorDataset(test_x, test_y),
                                                      batch_size=args['batch_size']*2,
                                                      shuffle=True, **kwargs)

        # linear
        model_l = LNet(emb_size, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_l.parameters(), lr=lr)

        for epoch in range(1, epochs+1):
            train(epoch, model_l, optimizer, train_loader_cls)
            correct = test2(model_l, test_loader_cls)

        scores[0][task_id][i][1] = correct/len(test_loader_cls.dataset)
        model_ls.append(model_l)
        
        # MLP
        model_mlp = MLP(emb_size, (emb_size+out_size[task_id])//2, out_size[task_id]).to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=lr)

        for epoch in range(1, epochs+1):
            train(epoch, model_mlp, optimizer, train_loader_cls)
            correct = test2(model_mlp, test_loader_cls)

        scores[0][task_id][i][2] = correct/len(test_loader_cls.dataset)
        model_mlps.append(model_mlp)
        
        # svm
        cls_svm = SVC(random_state=args['seed'])
        cls_svm.fit(data['train_x'], data['train_y'][task_id])
        _valid_score = cls_svm.score(data['test_x'], data['test_y'][task_id])
        scores[0][task_id][i][3] = _valid_score
        cls_svms.append(cls_svm)
        print(f'SVM test acc: {_valid_score:.5f}')

        # decision tree
        cls_dt = DecisionTreeClassifier(random_state=args['seed'])
        cls_dt.fit(data['train_x'], data['train_y'][task_id])
        _valid_score = cls_dt.score(data['test_x'], data['test_y'][task_id])
        scores[0][task_id][i][4] = _valid_score
        cls_dts.append(cls_dt)
        print(f'Decision Tree test acc: {_valid_score:.5f}')

        # random forest
        cls_rf = RandomForestClassifier(n_estimators=10, random_state=args['seed'], n_jobs=8)
        cls_rf.fit(data['train_x'], data['train_y'][task_id])
        _valid_score = cls_rf.score(data['test_x'], data['test_y'][task_id])
        scores[0][task_id][i][5] = _valid_score
        cls_rfs.append(cls_rf)
        print(f'Random Forest test acc: {_valid_score:.5f}')

        # lgb
        cls_lgb = LGBMClassifier(random_state=args['seed'], n_jobs=8)
        cls_lgb.fit(data['train_x'], data['train_y'][task_id],
                    eval_set=[(data['test_x'], data['test_y'][task_id])],
                    early_stopping_rounds=100,
                    verbose=100)
        _valid_pred = cls_lgb.predict(data['test_x'])
        _valid_score = sum(_valid_pred == data['test_y'][task_id])/len(_valid_pred)
        scores[0][task_id][i][6] = _valid_score
        cls_lgbs.append(cls_lgb)
        print(f'LightGBM test acc: {_valid_score:.5f}')

    # train the rest models with first model's CLS
    for j in range(1, n_model):
        print('train model:', j)
        start = j*step_size//2 if overlap else j*step_size
        end = start+step_size
        train_subset = Data.TensorDataset(train_x_all[start:end], train_y_all[start:end])
        train_loader = torch.utils.data.DataLoader(train_subset,
                                                   batch_size=args['batch_size'],
                                                   shuffle=True, **kwargs)
        model2 = VGG(emb_size).to(device)
        model2.cls = model1.cls
        model2.cls.weight.requires_grad = False
        model2.cls.bias.requires_grad = False
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, model2.parameters()),
                               lr=args['lr'],
                               momentum=0.9,
                               weight_decay=7e-4)

        for epoch in range(1, args['epochs']+51):
            lr = args['lr'] * (0.5**(epoch//30))
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr
            train(epoch, model2, optimizer2, train_loader)
            correct = test(model2, test_loader, task_id=0)
            if correct >= es_thd:
                print('Early stopping!')
                break

        for task_id in range(3):
            scores[j][task_id][i][0] = test(model2, test_loader, task_id=task_id)/len(test_loader.dataset)
        # save data for other CLSs
        print('extract embedding')
        emb_test = extract_embedding(model2, test_loader)
        data = {'test_x': np.array(emb_test['embedding']),
                'test_y': np.array(emb_test['target'])}

        # test CLSs
        print('test CLSs')
        for task_id in range(3):
            print('task:', task_id)
            # embedding
            test_x, test_y = torch.FloatTensor(data['test_x']), torch.LongTensor(data['test_y'][task_id])
            test_loader_cls = torch.utils.data.DataLoader(Data.TensorDataset(test_x, test_y),
                                                          batch_size=args['batch_size']*2,
                                                          shuffle=True, **kwargs)

            # linear
            correct = test2(model_ls[task_id], test_loader_cls)
            scores[j][task_id][i][1] = correct/len(test_loader_cls.dataset)

            # MLP
            correct = test2(model_mlps[task_id], test_loader_cls)
            scores[j][task_id][i][2] = correct/len(test_loader_cls.dataset)

            # svm
            _valid_score = cls_svms[task_id].score(data['test_x'], data['test_y'][task_id])
            scores[j][task_id][i][3] = _valid_score
            print(f'SVM test acc:{_valid_score:.5f}')

            # decision tree
            _valid_score = cls_dts[task_id].score(data['test_x'], data['test_y'][task_id])
            scores[j][task_id][i][4] = _valid_score
            print(f'Decision Tree test acc:{_valid_score:.5f}')

            # random forest
            _valid_score = cls_rfs[task_id].score(data['test_x'], data['test_y'][task_id])
            scores[j][task_id][i][5] = _valid_score
            print(f'Random Forest test acc:{_valid_score:.5f}')

            # lgb
            _valid_pred = cls_lgbs[task_id].predict(data['test_x'])
            _valid_score = sum(_valid_pred == data['test_y'][task_id])/len(_valid_pred)
            scores[j][task_id][i][6] = _valid_score
            print(f'LightGBM test acc:{_valid_score:.5f}')
    
    t = round(time.time()-t0)
    print('time consumed: {} min {} sec'.format(t//60, t%60))
    
    file_name = (
            "scores_incremental_overlap_v2_SGD.pkl"
            if overlap
            else "scores_incremental_v2_SGD.pkl"
        )
    pickle.dump(scores, open(file_name, 'wb'))