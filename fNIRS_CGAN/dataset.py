import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

class FNIRSDataset(Dataset):
    def __init__(self, data_rootdir):
        self.X, self.y = self.Load_Dataset(data_rootdir)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def Load_Dataset(data_rootdir):
        X_oxy = []
        y = []
        # 三分类标签
        class_labels = {'ADHD': 0, 'HC': 1, 'ASD': 2}
        for status in class_labels.keys():
            status_path = os.path.join(data_rootdir, status)
            for filename in os.listdir(status_path):
                filepath = os.path.join(status_path, filename)
                data = loadmat(filepath)
                nirsdata = data['nirsdata']['oxyData'].item(0)
                data = nirsdata[50:1550:]
                X_oxy.append(data)
                y.append(class_labels[status])
        X = np.array(X_oxy)
        y = np.array(y)
        print(f"X.shape:{X.shape}, y.shape:{y.shape}\n")
        return X, y 