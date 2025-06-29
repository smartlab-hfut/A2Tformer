import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def reindex_labels(y_train, y_test):
    # 在训练集和测试集中找出所有唯一的标签
    unique_labels = np.unique(np.concatenate((y_train, y_test)))

    # 创建从原始标签到从0开始的标签的映射
    label_mapping = {original_label: new_label for new_label, original_label in enumerate(unique_labels)}

    # 将映射应用到训练集和测试集
    y_train_reindexed = np.array([label_mapping[label] for label in y_train])
    y_test_reindexed = np.array([label_mapping[label] for label in y_test])

    return y_train_reindexed, y_test_reindexed

# 应用函数到您的标签




def prepare_data(data_name):
    filenameTSV1 = f'.\\data\\UCRArchive_2018\\UCRArchive_2018\\{data_name}\\{data_name}_TRAIN.tsv'
    filenameTSV2 = f'.\\data\\UCRArchive_2018\\UCRArchive_2018\\{data_name}\\{data_name}_TEST.tsv'

    x_train, y_train = readucr(filenameTSV1)
    x_test, y_test = readucr(filenameTSV2)

    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    x_train_standardized = scaler_standard.fit_transform(x_train)
    x_test_standardized = scaler_standard.transform(x_test)

    x_train_normalized = scaler_minmax.fit_transform(x_train_standardized)
    x_test_normalized = scaler_minmax.transform(x_test_standardized)

    y_train, y_test = reindex_labels(y_train, y_test)
    print(y_train, y_test)

    return x_train_normalized, y_train, x_test_normalized, y_test


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def get_loaders(x_train, y_train, x_test, y_test, batch_size):
    train_dataset = MyDataset(x_train, y_train)
    test_dataset = MyDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
