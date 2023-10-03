# Importing necessary packages
import scipy
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import pad

# Defining DEVICE
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bias=True,
            dropout=0.2,
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(11, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.RReLU(),
            nn.Linear(8, 4),
            nn.SiLU(),
            nn.Linear(4, 2),
            nn.SiLU(),
            nn.Linear(2, 1),
        )
            
        #TODO: use RELU6 next!
        
        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size + 9, hidden_size // 2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_size // 8),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_size // 8, output_size),
        )

    def forward(self, x, x_target):
        
        out = self.fc_1(x)
        out = out[..., 0]
        out, _ = self.lstm(out)
        out = out[:, -1, ...]
        out = torch.cat([out, x_target], dim=-1)
        out = self.fc_2(out)

        return out


class CustomDataset(Dataset):
    def __init__(
        self,
        path_X,
        path_x,
        path_y,
        upsample=False,
        distort_prob=0.0,
        smooth_labels=False,
        shuffle_neighbours=False,
    ):
        self.data_X = torch.load(path_X).to(dtype=torch.float32)
        self.data_X = self.data_X.permute(0, 1, 3, 2)

        self.data_x = torch.load(path_x).to(dtype=torch.float32)
        self.data_y = torch.load(path_y).to(dtype=torch.float32)

        if upsample is True:
            self.data_X, self.data_x, self.data_y = self.__upsample()

        self.distort_prob = distort_prob
        self.smooth_labels = smooth_labels
        self.shuffle_neighbours = shuffle_neighbours

    def __getitem__(self, index):
        X, x, y = (
            self.data_X[index].clone(),
            self.data_x[index].clone(),
            self.data_y[index].clone(),
        )

        if self.shuffle_neighbours is True:
            shuffled_idx = torch.randperm(10)
            neighbour_data = X[:, :, 1:]
            neighbour_data = neighbour_data[:, :, shuffled_idx]
            X = torch.cat([X[:, :, [0]], neighbour_data], dim=-1)

        if self.smooth_labels is True:
            y += (torch.rand(1).item() - 0.5) / (4 * 34)

        if torch.rand(1) < self.distort_prob:
            K = 2
            max_price = max(*X[:, -6, 0], x[-5])
            x[-5] = K * max_price
            y = torch.tensor([0], dtype=torch.float32)

        return X, x, y

    def __len__(self):
        return len(self.data_X)

    def __upsample(self, n_bins=10):
        MAX_K = 5

        min_value, max_value = self.data_y.min(), self.data_y.max()

        labels = torch.bucketize(
            boundaries=torch.linspace(min_value, max_value, n_bins),
            input=self.data_y,
            right=True,
        )
        bin_nums, counts = torch.unique(labels, return_counts=True)
        max_count = counts.max()

        X, x, y = [], [], []

        for bin_num, count in zip(bin_nums, counts):
            MASK = labels.flatten() == bin_num

            bin_entire_sample_X = self.data_X[MASK].clone()
            bin_entire_sample_x = self.data_x[MASK].clone()
            bin_entire_sample_y = self.data_y[MASK].clone()

            if count == max_count:
                X.append(bin_entire_sample_X)
                x.append(bin_entire_sample_x)
                y.append(bin_entire_sample_y)
                continue

            remaining_count = max_count - count

            if remaining_count < count:
                perm = torch.randperm(count)
                idx = perm[:remaining_count]

                sample_X = bin_entire_sample_X[idx].clone()
                sample_x = bin_entire_sample_x[idx].clone()
                sample_y = bin_entire_sample_y[idx].clone()

            else:
                k = torch.ceil(remaining_count / count).to(int).item()
                k = min(k, MAX_K)

                repeated_sample_X = bin_entire_sample_X.repeat(k, 1, 1, 1)
                repeated_sample_x = bin_entire_sample_x.repeat(k, 1)
                repeated_sample_y = bin_entire_sample_y.repeat(k, 1)

                perm = torch.randperm(k * count)
                idx = perm[:remaining_count]

                sample_X = repeated_sample_X[idx].clone()
                sample_x = repeated_sample_x[idx].clone()
                sample_y = repeated_sample_y[idx].clone()

            upsampled_X = torch.cat([bin_entire_sample_X, sample_X], dim=0)
            upsampled_x = torch.cat([bin_entire_sample_x, sample_x], dim=0)
            upsampled_y = torch.cat([bin_entire_sample_y, sample_y], dim=0)

            X.append(upsampled_X)
            x.append(upsampled_x)
            y.append(upsampled_y)

        X = torch.cat(X, dim=0)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        return X, x, y


def preprocess_data(qids, all_year_data, all_year_distances, data_2023):
    MAX_NEIGH = 10
    columns_to_drop = ["PPSVACWert", *[f"T{i}" for i in range(1, 35)], "target"]

    X, x, y = [], [], []

    for idx, qid in enumerate(tqdm(qids)):
        neighbours_features = []

        for year_data, year_distances in zip(all_year_data, all_year_distances):
            current_distances = year_distances[year_distances.index == qid]

            if current_distances.shape[0] != 0:
                current_year_neighbours = current_distances[
                    current_distances.Qid2 != qid
                ]
                current_year_neighbours_data = torch.from_numpy(
                    year_data.loc[current_year_neighbours.Qid2].values
                )
                current_year_neighbours_data_padded = pad(
                    current_year_neighbours_data,
                    (0, 0, 0, MAX_NEIGH - current_year_neighbours_data.shape[0]),
                    "constant",
                    0,
                )

                current_year_self_data = torch.from_numpy(year_data.loc[qid].values)

                if (current_year_self_data.ndim == 2) and (
                    current_year_self_data.shape[0] > 1
                ):
                    current_year_self_data = current_year_self_data[0]

                current_year_data_point = torch.cat(
                    [current_year_self_data[None], current_year_neighbours_data_padded],
                    dim=0,
                )

            else:
                current_year_data_point = torch.zeros(MAX_NEIGH + 1, year_data.shape[1])

            neighbours_features.append(current_year_data_point)

        self_data_2023 = torch.from_numpy(
            data_2023.loc[qid].drop(labels=columns_to_drop).values
        )
        neighbours_features = torch.stack(neighbours_features, dim=0)
        label = torch.tensor([data_2023.loc[qid, "target"]])

        X.append(neighbours_features)
        x.append(self_data_2023)
        y.append(label)

    X = torch.stack(X, dim=0)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)

    return X, x, y


def train(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for batch in tqdm(data_loader, position=0, leave=True):
        X_batch = batch[0].to(DEVICE)
        x_batch = batch[1].to(DEVICE)
        y_batch = batch[2].to(DEVICE)
        outputs = model(X_batch, x_batch)
        loss = criterion(y_batch, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    return running_loss / len(data_loader)


def validate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, position=0, leave=True):
            X_batch = batch[0].to(DEVICE)
            x_batch = batch[1].to(DEVICE)
            y_batch = batch[2].to(DEVICE)
            outputs = model(X_batch, x_batch)
            loss = criterion(y_batch, outputs)
            running_loss += loss.item()

    return running_loss / len(data_loader)
