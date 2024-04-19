import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):

        self.data = dataframe
        self.transform = transform

        # Extracting relevant columns
        self.features = self.data[
            ["OpenInterest", "Delta", "Gamma", "Theta", "Vega"]
        ].values.astype(float)
        self.target = (self.data["TargetPrice"] - self.data["BSAprox"]).values.astype(
            float
        )
        # normalize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "input": torch.tensor(self.features[idx], dtype=torch.float),
            "output": torch.tensor(self.target[idx], dtype=torch.float),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class CustomBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0):
        super(CustomBlock, self).__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class Call_1(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(Call_1, self).__init__()
        self.fcs1 = nn.Sequential(
            # initial layer
            CustomBlock(N_INPUT, N_HIDDEN, dropout_prob=0),
            # middle layers
            *[CustomBlock(N_HIDDEN, N_HIDDEN) for _ in range(N_LAYERS)],
            # last layer with 1 output
            nn.Linear(N_HIDDEN, N_OUTPUT)
        )

    def forward(self, x):
        x = self.fcs1(x)
        return x


class Call_2(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(Call_2, self).__init__()

        # Define the initial layer
        self.initial_layer = nn.Sequential(
            CustomBlock(N_INPUT, 128, dropout_prob=0),
            CustomBlock(128, 256, dropout_prob=0),
            CustomBlock(256, N_HIDDEN, dropout_prob=0),
        )

        # Define the middle layers with skip connections
        self.middle_layers = nn.ModuleList(
            [CustomBlock(N_HIDDEN, N_HIDDEN) for _ in range(N_LAYERS)]
        )

        # Define the last layer with 1 output
        self.last_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        # Pass through the initial layer
        x = self.initial_layer(x)

        # Pass through the middle layers with skip connections
        for layer in self.middle_layers:
            # Apply the layer
            out = layer(x)
            # Add skip connection
            x = x + out

        # Pass through the last layer
        x = self.last_layer(x)

        return x


class Put_1(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(Put_1, self).__init__()
        self.fcs1 = nn.Sequential(
            # initial layer
            CustomBlock(N_INPUT, N_HIDDEN, dropout_prob=0),
            # middle layers
            *[CustomBlock(N_HIDDEN, N_HIDDEN) for _ in range(N_LAYERS)],
            # last layer with 1 output
            nn.Linear(N_HIDDEN, N_OUTPUT)
        )

    def forward(self, x):
        x = self.fcs1(x)
        return x


class Put_2(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(Put_2, self).__init__()

        # Define the initial layer
        self.initial_layer = nn.Sequential(
            CustomBlock(N_INPUT, 128, dropout_prob=0),
            CustomBlock(128, 256, dropout_prob=0),
            CustomBlock(256, N_HIDDEN, dropout_prob=0),
        )

        # Define the middle layers with skip connections
        self.middle_layers = nn.ModuleList(
            [CustomBlock(N_HIDDEN, N_HIDDEN) for _ in range(N_LAYERS)]
        )

        # Define the last layer with 1 output
        self.last_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        # Pass through the initial layer
        x = self.initial_layer(x)

        # Pass through the middle layers with skip connections
        for layer in self.middle_layers:
            # Apply the layer
            out = layer(x)
            # Add skip connection
            x = x + out

        # Pass through the last layer
        x = self.last_layer(x)

        return x


class AmericanPut_gated3(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(AmericanPut_gated3, self).__init__()
        self.N_HIDDEN = N_HIDDEN
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.Tanh()
        self.fcs1 = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), self.activation2)
        self.fcs2 = nn.Sequential(nn.Linear(N_INPUT, N_OUTPUT), self.activation2)
        self.fch = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), self.activation2)
                for _ in range(N_LAYERS)
            ]
        )
        self.fce = nn.Sequential(*[nn.Linear(N_HIDDEN, N_OUTPUT)])
        self.w1_layer = nn.Sequential(
            *[nn.Linear(N_HIDDEN + N_INPUT, N_OUTPUT), self.activation2]
        )
        # self.w2_layer = nn.Sequential(*[nn.Linear(N_HIDDEN + N_INPUT, N_OUTPUT),self.activation2])

    def forward(self, x):
        # Apply the first layer
        I1 = self.fcs1(x)
        H = I1
        # Apply hidden layers with residual connections
        for layer in self.fch:
            H = layer(H) + H
        # Apply the final layer
        yx = self.fcs2(x)  # 1D
        yh = self.fce(H)  # 1D
        h_x = torch.cat([H, x], axis=1)
        # print (h_x.shape)
        wh = self.w1_layer(h_x)
        y_net = yx + wh * yh
        return y_net
