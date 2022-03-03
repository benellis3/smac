from black import out
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from smac.classifier.dataloader import ObsAndStatesDataset


class FFNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(
            self.in_features, out_features=self.hidden_features
        )
        self.fc2 = nn.Linear(
            self.hidden_features, out_features=self.hidden_features
        )
        self.fc_mean = nn.Linear(
            in_features=self.hidden_features, out_features=self.out_features
        )
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_mean(x)


def get_features(dataset):
    obs, state = dataset[0]
    return obs.shape[0], state.shape[0]


@hydra.main(config_path="conf", config_name="config")
def main(cfg: OmegaConf):

    # make the dataloader
    cfg.device = "cuda" if th.cuda.is_available() else "cpu"
    dataset = ObsAndStatesDataset(cfg.dataset_path)
    dataloader = DataLoader(
        dataset=dataset, shuffle=True, batch_size=cfg.batch_size
    )
    # make the module
    in_features, out_features = get_features(dataset)
    ffnet = FFNet(in_features, out_features, cfg.hidden_features, cfg.device)
    ffnet.to(cfg.device)
    # make the optimiser
    optimiser = Adam(params=ffnet.parameters(), lr=cfg.lr)
    loss = nn.MSELoss(reduction="sum")
    # iterate through the dataloader
    for epoch in range(cfg.epochs):
        size = len(dataloader.dataset)
        cum_loss = 0.0
        for obs, states in tqdm(dataloader):
            pred_states = ffnet(obs)
            loss_val = loss(pred_states, states)
            cum_loss += loss_val
            optimiser.zero_grad()
            loss_val.backward()
            optimiser.step()

        print(f"Epoch {epoch}: {cum_loss / size}")


if __name__ == "__main__":
    main()
