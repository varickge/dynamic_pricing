import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from torch.nn.functional import pad

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
else:
    DEVICE = torch.device("cpu")


class LSTMModel_1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel_1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bias=True,
            dropout=0.0,
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(11, 1),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_size // 1, hidden_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout1d(0.0),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Dropout1d(0.0),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.LeakyReLU(),
            nn.Dropout1d(0.0),
            nn.Linear(hidden_size // 8, output_size),
        )

        self.fc_b = nn.Linear(input_size - 69, input_size)

    def forward(self, x, x_2023):
        out = self.fc_1(x)
        out = out[..., 0]

        x_2023 = self.fc_b(x_2023)[:, None, ...]
        out = torch.cat([out, x_2023], dim=1)

        out, _ = self.lstm(out)
        out = out[:, -1, ...]
        out = self.fc_2(out)

        return out


class PriceOptimizer:
    def __init__(
        self,
        data_root_dir,
        out_root_dir,
        price_min,
        price_max,
        price_num,
    ):
        data_root_dir = Path(data_root_dir)

        if not data_root_dir.is_dir():
            raise ValueError(
                f"The directory {data_root_dir.as_posix()} does not exist."
            )

        out_root_dir = Path(out_root_dir)

        if not out_root_dir.is_dir():
            raise ValueError(f"The directory {out_root_dir.as_posix()} does not exist.")

        if not isinstance(price_min, (int, float)):
            raise TypeError(
                f"Expected price_min to be type of int or float, but got {type(price_min)}."
            )

        if not isinstance(price_max, (int, float)):
            raise TypeError(
                f"Expected price_max to be type of int or float, but got {type(price_max)}."
            )

        if not isinstance(price_num, int):
            raise TypeError(
                f"Expected price_num to be type of int, but got {type(price_num)}."
            )

        if price_max <= price_min:
            raise ValueError(f"Inconsistent price range.")

        if not (10 <= price_num <= 2000):
            raise ValueError(f"Inconsistent price_num.")

        self.out_dir = out_root_dir
        self.data_root_dir = data_root_dir
        self.price_min = price_min
        self.price_max = price_max
        self.price_num = price_num

        # read the data

        data_path = data_root_dir / "data.pkl"

        if not data_path.is_file():
            raise ValueError(f"The file {data_path.as_posix()} does not exist.")

        self.data = pd.read_pickle(data_path)

        # calculate price mean and std
        self.price_mean = self.data.Preis.mean()
        self.price_std = self.data.Preis.std()

        # load model
        self.nn = LSTMModel_1(78, 512, 5, 1).to(device=DEVICE).eval()
        self.nn.load_state_dict(torch.load("./final_data/dense_lstm_model.pt"))

        # preprocess the data
        self.__preprocess_data()

        # load distance data
        self.__load_distance_data()

    def __load_distance_data(self):
        self.distance_2018 = pd.read_csv(
            self.data_root_dir / "distances_2018.csv"
        ).set_index("Qid_1")
        self.distance_2019 = pd.read_csv(
            self.data_root_dir / "distances_2019.csv"
        ).set_index("Qid_1")
        self.distance_2020 = pd.read_csv(
            self.data_root_dir / "distances_2020.csv"
        ).set_index("Qid_1")
        self.distance_2021 = pd.read_csv(
            self.data_root_dir / "distances_2021.csv"
        ).set_index("Qid_1")
        self.distance_2022 = pd.read_csv(
            self.data_root_dir / "distances_2022.csv"
        ).set_index("Qid_1")

    def __normalize_columns(self):
        self.cleaned_data.Preis = (
            self.cleaned_data.Preis - self.cleaned_data.Preis.mean()
        ) / self.cleaned_data.Preis.std()
        self.cleaned_data.Laenge = (
            self.cleaned_data.Laenge - self.cleaned_data.Laenge.mean()
        ) / self.cleaned_data.Laenge.std()
        self.cleaned_data.Breite = (
            self.cleaned_data.Breite - self.cleaned_data.Breite.mean()
        ) / self.cleaned_data.Breite.std()
        self.cleaned_data.PPSVACWert = (
            self.cleaned_data.PPSVACWert - self.cleaned_data.PPSVACWert.mean()
        ) / self.cleaned_data.PPSVACWert.std()

        list_of_t = [f"TD{i:02d}" for i in range(1, 35)]

        for i in list_of_t:
            self.cleaned_data[i] = (
                self.cleaned_data[i] - self.cleaned_data[i].mean()
            ) / self.cleaned_data[i].std()

    def __separate_data(self):
        self.data_2018 = (
            self.cleaned_data[self.cleaned_data.GJ == 2018]
            .drop(columns=["Qid", "GJ"])
            .copy()
        )
        self.data_2019 = (
            self.cleaned_data[self.cleaned_data.GJ == 2019]
            .drop(columns=["Qid", "GJ"])
            .copy()
        )
        self.data_2020 = (
            self.cleaned_data[self.cleaned_data.GJ == 2020]
            .drop(columns=["Qid", "GJ"])
            .copy()
        )
        self.data_2021 = (
            self.cleaned_data[self.cleaned_data.GJ == 2021]
            .drop(columns=["Qid", "GJ"])
            .copy()
        )
        self.data_2022 = (
            self.cleaned_data[self.cleaned_data.GJ == 2022]
            .drop(columns=["Qid", "GJ"])
            .copy()
        )
        self.data_2023 = (
            self.cleaned_data[self.cleaned_data.GJ == 2023]
            .drop(columns=["Qid", "GJ"])
            .copy()
        )

    def __preprocess_data(self):
        CONST_COLUMNS = [
            "Stellensubart_1",
            "Stellensubart_2",
            "Stellensubart_3",
            "Stellensubart_4",
            *[f"T{i}" for i in range(1, 35)],
            *[f"TD{i:02d}" for i in range(1, 35)],
            "Preis",
            "Beleuchtet",
            "Laenge",
            "Breite",
            "Eigenfläche",
            "PPSVACWert",
            "Qid",
            "GJ",
        ]

        # set Qid column as index and take the required columns
        cleaned_data = self.data.set_index("Qid", drop=False)
        cleaned_data.Qid = cleaned_data.Qid.astype(int)
        self.cleaned_data = cleaned_data.loc[:, CONST_COLUMNS]

        # normalize the data
        self.__normalize_columns()

        # separate the data into years
        self.__separate_data()

    def __get_qid_data(self, qid):
        if (self.cleaned_data.index == qid).sum() == 0:
            raise ValueError(f"Unknown qid {qid}")

        if (self.data_2023.index == qid).sum() == 0:
            raise ValueError(f"There is no information for qid {qid} for 2023 year.")
        MAX_NEIGH = 10

        # columns which will be dropped for target year (2023)
        columns_to_drop = [
            "PPSVACWert",
            *[f"T{i}" for i in range(1, 35)],
            *[f"TD{i:02d}" for i in range(1, 35)],
        ]

        neighbours_features = []

        all_year_data = [
            self.data_2018,
            self.data_2019,
            self.data_2020,
            self.data_2021,
            self.data_2022,
        ]

        all_year_distances = [
            self.distance_2018,
            self.distance_2019,
            self.distance_2020,
            self.distance_2021,
            self.distance_2022,
        ]

        for year_data, year_distances in zip(all_year_data, all_year_distances):
            current_distances = year_distances[year_distances.index == qid]

            if (current_distances.shape[0] != 0) and (
                (year_data.index == qid).sum() != 0
            ):
                current_year_neighbours = current_distances[
                    current_distances.Qid_2 != qid
                ]

                current_year_neighbours_data = torch.from_numpy(
                    year_data[
                        year_data.index.isin(current_year_neighbours.Qid_2)
                    ].values
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
                current_year_data_point = torch.zeros(11, 78)

            neighbours_features.append(current_year_data_point)

        self_data_2023 = torch.from_numpy(
            self.data_2023.loc[qid].drop(labels=columns_to_drop).values
        )

        neighbours_features = torch.stack(neighbours_features, dim=0)
        label = torch.tensor(self.data_2023.loc[qid, "T1":"T22"].replace(-1, np.nan).mean())
        
        return neighbours_features, self_data_2023

    def __grid_search(self, qid_data):
        data_X, data_x = qid_data

        data_X = data_X[None].permute(0, 1, 3, 2).to(dtype=torch.float32, device=DEVICE)
        data_x = data_x[None].to(dtype=torch.float32, device=DEVICE)

        history = np.zeros((3, self.price_num))

        # denormalize the price
        original_price = data_x[:, -5].item() * self.price_std + self.price_mean

        data_X = data_X.repeat(self.price_num, 1, 1, 1)
        data_x = data_x.repeat(self.price_num, 1)

        price_grid = torch.linspace(self.price_min, self.price_max, self.price_num)
        data_x[:, -5] = (price_grid - self.price_mean) / self.price_std

        mean_b = self.nn(data_X, data_x).detach().cpu().numpy()[..., 0]
        reward = price_grid * mean_b
        history[0] = price_grid
        history[1] = mean_b
        history[2] = reward

        return original_price, history

    def __find_optimum(self, history, original_price):
        def find_nearest(array, value, idxis=False):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            if idxis:
                return idx
            return array[idx], idx
        
        def find_volume(array, peak, value, idxis=False):
            # find nearest indices and values around 'peak'to 95% of reward
            array = np.asarray(array)
            idx=[]
            idx.append((np.abs(array[:peak] - value)).argsort()[:1])
            idx.append((np.abs(array[peak:] - value)).argsort()[:1]+peak)
            return array[idx], idx

        # find the nearest peak
        peak_ids = scipy.signal.find_peaks(history[2])[0]

        # if no peak appears, return nan
        if len(peak_ids) == 0:
            return float("nan"), float("nan")

        prices = history[0, peak_ids]
        optimal_price, idx = find_nearest(prices, original_price)
        reward_95 =  history[2, peak_ids[idx]]*0.95
        volume, volume_idx = find_volume(history[2], peak_ids[idx], reward_95) 

        min_price = history[0, volume_idx][0].item()
        max_price = history[0, volume_idx][1].item()
        mean_booking = history[1, peak_ids[idx]]
        return optimal_price, min_price, max_price

    def __visalize(self, history, optimal_price, original_price, min_price, max_price, qid, root):
        fig, ax = plt.subplots()
        ax.grid(True)

        ax.plot(history[0], history[-1], color="blue")

        if not math.isnan(optimal_price):
            ax.axvline(x=original_price, color="green")
            ax.axvline(x=optimal_price, color="red")
            ax.legend(["reward curve", "original price", "optimal price"])
            ax.axvline(x=min_price, color="red", ls="--")
            ax.axvline(x=max_price, color="red", ls="--")

        else:
            ax.axvline(x=original_price, color="green")
            ax.legend(["original price", "reward curve"])

        fig.tight_layout()
        fig.savefig(root / f"reward_plot_qid_{qid}.png", dpi=300)

    def __make_dataframe(self, qid, original_price, optimal_price, min_price, max_price, root):
        """
        Creates a pandas DataFrame from inputs and saves as csv file.

            Parameters:
                qid (int): Ad qid
                original_price (float): Ad original price
                optimal_price (float): Ad optimal price
                min_price (float): Price corresponding to 95% reward less than the optimal price
                max_price (float): Price corresponding to 95% reward greather than the optimal price
                root (str, Path): Root path for csv file.

            Returns:
                None
        """
        columns = [
            "Qid",
            "Original price",
            "Optimal price",
            "Min_optimal_preis_5%",
            "Max_optimal_preis_5%_margin",
        ]
        data = np.array(
            [
                [
                    qid,
                    original_price,
                    optimal_price,
                    min_price, 
                    max_price,
                ]
            ]
        )
        df = pd.DataFrame(columns=columns, data=data)
        df.Qid = df.Qid.astype(int)
        df.to_csv(root / f"result_qid_{qid}.csv", index=False)

    def __call__(self, qid: int):
        """
        Return the optimal price and the 5 percent margin for given `qid`.

            Parameters:
                qid (int): Ad qid

            Returns:
                result (tuple(float, float)): Ad optimal price and margin.
                If the optimum does not exist, returns (nan, nan)
        """
        if not isinstance(qid, int):
            raise TypeError(f"Expected qid to be type of int, but got {type(qid)}.")

        if qid < 0:
            raise ValueError(
                f"Expected qid to be greather or equal to zero, but got {qid}"
            )

        # get data for qid to pass into nn
        qid_data = self.__get_qid_data(qid)

        # find optimal price and margin
        original_price, history = self.__grid_search(qid_data)
        optimal_price, min_price, max_price = self.__find_optimum(history, original_price)

        self.__visalize(
            history, optimal_price, original_price, min_price, max_price, qid, self.out_dir
        )
        self.__make_dataframe(qid, original_price, optimal_price, min_price, max_price, self.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root-dir", help="The root path for data.", type=str, required=True
    )
    parser.add_argument(
        "--out-root-dir",
        help="Path for saving output files.",
        default="./",
        type=str,
        required=True,
    )
    parser.add_argument("--qid", help="Inference qid.", type=int, required=True)
    parser.add_argument(
        "--price-min", help="The minimum value for price.", default=0, type=float
    )
    parser.add_argument(
        "--price-max", help="The maximum value for price.", default=100, type=float
    )
    parser.add_argument(
        "--price-num",
        help="The number of samples between min and max",
        default=1000,
        type=int,
    )

    args = parser.parse_args()

    optimizer = PriceOptimizer(
        args.data_root_dir,
        args.out_root_dir,
        args.price_min,
        args.price_max,
        args.price_num,
    )

    optimizer(args.qid)
