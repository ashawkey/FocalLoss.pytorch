import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

Device = torch.device("cuda")
Epoch = 32

np.random.seed(42)

class fcdr(nn.Module):
    def __init__(self, Fin, Fout, dp=0.5):
        super(fcdr, self).__init__()
        self.fc = nn.Linear(Fin, Fout)
        self.dp = nn.Dropout(dp)
        self.ac = nn.ReLU(True)
    def forward(self, x):
        x = self.fc(x)
        x = self.dp(x)
        x = self.ac(x)
        return x

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc0 = fcdr(10, 256)
        self.fc1 = fcdr(256, 512) 
        self.fc2 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x) # [B, 2]
        return x

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class Averager:
    """ statistics for classification """
    def __init__(self, nCls=2):
        self.nCls = nCls
        self.N = 0
        self.eps = 1e-15
        self.table = np.zeros((nCls, 4), dtype = np.int32)

    def update(self, logits, truth):
        self.N += 1
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy() # [B, N]
        labels = truth.detach().cpu().numpy() # [B, ]
        for Cls in range(self.nCls):
            true_positive = np.count_nonzero(np.bitwise_and(preds == Cls, labels == Cls))
            true_negative = np.count_nonzero(np.bitwise_and(preds != Cls, labels != Cls))
            false_positive = np.count_nonzero(np.bitwise_and(preds == Cls, labels != Cls))
            false_negative = np.count_nonzero(np.bitwise_and(preds != Cls, labels == Cls))
            self.table[Cls] += [true_positive, true_negative, false_positive, false_negative]
    
    def measure(self):
        precisions = []
        recalls = []
        for Cls in range(self.nCls):
            precision = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,3] + self.eps) # TP / (TP + FN)
            recall = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,2] + self.eps) # TP / (TP + FP)
            precisions.append(precision)
            recalls.append(recall)
        total_TP = np.sum(self.table[:, 0]) # all true positives 
        total = np.sum(self.table[0]) # total trials
        accuracy = total_TP/total
        return accuracy, precisions, recalls

    def report(self, intro, multiclass=True):
        accuracy, precisions, recalls = self.measure()
        text = "{}: Accuracy = {:.4f}\n".format(intro, accuracy)
        if multiclass:
            for Cls in range(self.nCls):
                text += "\tClass {}: precision = {:.3f} recall = {:.3f}\n".format(Cls, precisions[Cls], recalls[Cls])
        print(text, end='')

# non-batched
def train(X, Y, model, criterion, optimizer, epoch):
    model.train()
    avg = Averager()
    for x, y in zip(X, Y):
        x = torch.from_numpy(x).to(Device)
        y = torch.from_numpy(y).to(Device)
        preds = model(x).unsqueeze(0) # [1, 2] for [N, C], due to non-batch 
        loss = criterion(preds, y) # [1] for [N, ]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg.update(preds, y)
    avg.report("==> Epoch {}".format(epoch))
        
def validate(X, Y, model, epoch):
    model.eval()
    avg = Averager()
    for x, y in zip(X, Y):
        x = torch.from_numpy(x).to(Device)
        y = torch.from_numpy(y).to(Device)
        with torch.no_grad():
            preds = model(x).unsqueeze(0)
        avg.update(preds, y)
    avg.report("++> Validate {}".format(epoch))

def gendata(N, p=[0.9, 0.1]):
    Y = np.random.choice(2, size=(N, 1), p=p).astype(np.int64)
    X = (np.random.rand(N, 10)+Y*0.2).astype(np.float32)
    #X = np.hstack((Y, X)).astype(np.float32)
    #Y = np.bitwise_xor(Y, np.ones_like(Y)) # reverse it
    return X, Y

if __name__ == "__main__":
    train_data = gendata(1000)
    val_data = gendata(500)
    model = FCN().to(Device)
    criterion = FocalLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(Epoch):
        train(train_data[0], train_data[1], model, criterion, optimizer, epoch)
        validate(val_data[0], val_data[1], model, epoch)

