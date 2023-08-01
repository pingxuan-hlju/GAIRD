from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, data, associations):
        self.data = data
        self.associations = associations

    def __getitem__(self, idx):
        x, y = self.data[idx]
        label = self.associations[x][y]
        return x, y, label

    def __len__(self):
        return len(self.data)
