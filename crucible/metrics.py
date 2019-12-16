import os
import torch
import numpy as np

"""
How to add new metrics:
    * define a function that accepts numpy (pred, truth) and returns scalar/vector metric
    * wrap it with ScalarMeter/VectorMeter
"""


def IoU(preds, truths):
    """
    preds: [B, H, W] 
    truths: [B, H, W]
    """
    batch_size = truths.shape[0]
    ious = []

    for batch in range(batch_size):
        for part in range(self.nCls):
            I = np.sum(np.logical_and(preds[batch] == part, truths[batch] == part))
            U = np.sum(np.logical_or(preds[batch] == part, truths[batch] == part))
            if U == 0: 
                continue
            else:
                ious.append(I/U)
    
    return np.mean(ious)

class ScalarMeter:
    def __init__(self, name, core, larger=True, reduction="mean"):
        self.core = core
        self.data = []
        self.larger = larger
        self.reduction = reduction
        self.name = name
    
    def clear(self):
        self.data = []

    def prepare_inputs(self, outputs, truths):
        """
        outputs and truths are pytorch tensors or numpy ndarrays.
        """
        if torch.is_tensor(outputs):
            outputs = outputs.detach().cpu().numpy()
        if torch.is_tensor(truths):
            truths = truths.detach().cpu().numpy()
        
        return outputs, truths

    def update(self, outputs, truths):
        outputs, truths = self.prepare_inputs(outputs, truths)
        res = self.core(outputs, truths)
        self.data.append(res)

    def measure(self):
        if self.reduction == "mean":
            return np.mean(self.data)
        elif self.reduction == "sum":
            return np.sum(self.data)
    
    def better(self, A, B):
        if self.larger:
            return A > B
        else:
            return A < B

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, self.name), self.measure(), global_step)

    def report(self):
        text = f"{self.name} = {self.measure():.4f}\n"
        return text

class VectorMeter:
    def __init__(self, name, core, larger=True, reduction="mean"):
        self.core = core
        """
        core: lambda outputs, truths -> [vector]
            assume core function accepts torch.Tensor
            Classification needs three meters: accuracy, precision, recall

        larger: True
            the larger, the better
        """
        self.data = []
        self.larger = larger
        self.reduction = reduction
        self.name = name
    
    def clear(self):
        self.data = []

    def prepare_inputs(self, outputs, truths):
        """
        outputs and truths are pytorch tensors or numpy ndarrays.
        """
        if torch.is_tensor(outputs):
            outputs = outputs.detach().cpu().numpy()
        if torch.is_tensor(truths):
            truths = truths.detach().cpu().numpy()
        
        return outputs, truths

    def update(self, outputs, truths):
        outputs, truths = self.prepare_inputs(outputs, truths)
        res = self.core(outputs, truths)
        self.data.append(res)

    def measure(self):
        if self.reduction == "mean":
            return np.mean(self.data)
        elif self.reduction == "sum":
            return np.sum(self.data)
    
    def better(self, A, B):
        if self.larger:
            return A > B
        else:
            return A < B

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, self.name), self.measure(), global_step)

    def report(self):
        res = np.mean(self.data, axis=0)
        text = f"{self.name}: mean = {np.mean(res):.4f}\n"
        for i in range(len(res)):
            text += f"\tClass {i} = {res[i]:.4f}\n"
        return text

class ClassificationMeter:
    """ statistics for classification """
    def __init__(self, nCls, eps=1e-5, names=None, keep_history=False):
        self.nCls = nCls
        self.names = names
        self.eps = eps
        self.N = 0
        self.table = np.zeros((self.nCls, 4), dtype=np.int32)
        self.keep_history = keep_history
        if keep_history:
            self.hist_preds = []
            self.hist_truths = []

    def clear(self):
        self.N = 0
        self.table = np.zeros((self.nCls, 4), dtype=np.int32)
        if self.keep_history:
            self.hist_preds = []
            self.hist_truths = []

    def prepare_inputs(self, outputs, truths):
        """
        outputs and truths are pytorch tensors or numpy ndarrays.
        """
        if torch.is_tensor(outputs):
            outputs = outputs.detach().cpu().numpy()
        if torch.is_tensor(truths):
            truths = truths.detach().cpu().numpy()
        
        return outputs, truths

    def update(self, preds, truths):

        preds, truths = self.prepare_inputs(preds, truths)

        if self.keep_history:
            self.hist_preds.extend(preds.tolist())
            self.hist_truths.extend(truths.tolist())

        self.N += np.prod(truths.shape)
        for Cls in range(self.nCls):
            true_positive = np.count_nonzero(np.bitwise_and(preds == Cls, truths == Cls))
            true_negative = np.count_nonzero(np.bitwise_and(preds != Cls, truths != Cls))
            false_positive = np.count_nonzero(np.bitwise_and(preds == Cls, truths != Cls))
            false_negative = np.count_nonzero(np.bitwise_and(preds != Cls, truths == Cls))
            self.table[Cls] += [true_positive, true_negative, false_positive, false_negative]

    def measure(self):
        """Overall Accuracy"""
        total_TP = np.sum(self.table[:, 0]) # all true positives 
        accuracy = total_TP/self.N
        return accuracy

    def better(self, A, B):
        return A > B

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "Accuracy"), self.measure(), global_step)
    
    def plot_conf_mat(self):
        if not self.keep_history:
            print("[ERROR]: classification meter not keeping history.")
            return
        #mat = confusion_matrix(self.hist_truths, self.hist_preds)
        from .vision import plot_confusion_matrix
        plot_confusion_matrix(self.hist_truths, self.hist_preds)

    def report(self, each_class=True, conf_mat=False):
        precisions = []
        recalls = []
        for Cls in range(self.nCls):
            precision = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,3] + self.eps) # TP / (TP + FN)
            recall = self.table[Cls,0] / (self.table[Cls,0] + self.table[Cls,2] + self.eps) # TP / (TP + FP)
            precisions.append(precision)
            recalls.append(recall)
        total_TP = np.sum(self.table[:, 0]) # all true positives 
        accuracy = total_TP/self.N
        accuracy_mean_class = np.mean(precisions)

        text = f"Overall Accuracy = {accuracy:.4f}({total_TP}/{self.N})\n"
        text += f"\tMean-class Accuracy = {accuracy_mean_class:.4f}\n"
        
        if each_class:
            for Cls in range(self.nCls):
                #if precisions[Cls] != 0 or recalls[Cls] != 0:
                text += f"\tClass {str(Cls)+'('+self.names[Cls]+')' if self.names is not None else Cls}: precision = {precisions[Cls]:.3f} recall = {recalls[Cls]:.3f}\n"
        if conf_mat:
            self.plot_conf_mat()

        return text
