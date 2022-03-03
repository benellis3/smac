from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch as th
from typing import Union


class ObsAndStatesDataset(Dataset):
    def __init__(self, path: Union[Path, str]):
        super().__init__()
        self.path = Path(path) if not isinstance(path, Path) else path
        self._load_data()

    def _load_data(self):
        self.states = []
        self.obs = []
        for pth in [f for f in self.path.iterdir() if f.is_file()]:
            if "state" in pth.name:
                self.states.append(np.load(pth))
            elif "obs" in pth.name:
                self.obs.append(np.load(pth, allow_pickle=True))

        self.obs = np.concatenate(self.obs)
        self.states = np.concatenate(self.states)
        self.states = np.expand_dims(self.states, axis=1)
        self.states = np.repeat(self.states, self.obs.shape[1], axis=1)
        self.obs = np.reshape(self.obs, (-1, self.obs.shape[2]))
        self.states = np.reshape(self.states, (-1, self.states.shape[2]))

    def __len__(self):
        assert self.obs.shape[0] == self.states.shape[0]
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return th.tensor(self.obs[idx]), th.tensor(self.states[idx])


if __name__ == "__main__":
    ds = ObsAndStatesDataset(
        Path(__file__).parent.parent.parent / "supervised_obs_and_states"
    )

    print(ds[0])
