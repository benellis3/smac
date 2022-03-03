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
        self.fc3 = nn.Linear(
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
        x = F.relu(self.fc3(x))
        return self.fc_mean(x)


def get_features(dataset):
    obs, state = dataset[0]
    return obs.shape[0], state.shape[0]


def train(dataloader, model, optimiser, loss_fn, device):
    for obs, states in tqdm(dataloader):
        obs = obs.to(device)
        states = states.to(device)
        pred_states = model(obs)
        loss_val = loss_fn(pred_states, states)
        optimiser.zero_grad()
        loss_val.backward()
        optimiser.step()


def test(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    no_close_items = dataloader.dataset[0][1].shape[0] * size
    test_loss = 0
    close_items = 0
    metrics = {}
    with th.no_grad():
        for obs, states in tqdm(dataloader):
            obs = obs.to(device)
            states = states.to(device)
            pred_states = model(obs)
            test_loss += loss_fn(pred_states, states).item()
            close_items += (
                th.isclose(pred_states, states, rtol=1e-2, atol=1e-5)
                .type(th.float)
                .sum()
                .item()
            )
    metrics["test_loss"] = test_loss / size
    metrics["close_items"] = close_items / no_close_items
    model.train()
    print(metrics)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: OmegaConf):

    # make the dataloader
    cfg.device = "cuda" if th.cuda.is_available() else "cpu"
    dataset = ObsAndStatesDataset(cfg.dataset_path)
    dataloader = DataLoader(
        dataset=dataset, shuffle=True, batch_size=cfg.batch_size
    )
    test_dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size)
    # make the module
    in_features, out_features = get_features(dataset)
    ffnet = FFNet(in_features, out_features, cfg.hidden_features, cfg.device)
    ffnet.to(cfg.device)
    # make the optimiser
    optimiser = Adam(params=ffnet.parameters(), lr=cfg.lr)
    loss = nn.MSELoss(reduction="mean")
    # iterate through the dataloader
    for epoch in range(cfg.epochs):
        train(
            dataloader=dataloader,
            model=ffnet,
            optimiser=optimiser,
            loss_fn=loss,
            device=cfg.device,
        )
        if epoch % cfg.test_frequency == 0:
            test(
                dataloader=test_dataloader,
                model=ffnet,
                loss_fn=loss,
                device=cfg.device,
            )


if __name__ == "__main__":
    main()
