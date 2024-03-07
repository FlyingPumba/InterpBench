import numpy as np
from torch.utils.data import Dataset, DataLoader
from iit.utils.config import DEVICE
import torch
from iit.utils.iit_dataset import IITDataset


class TracrDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels
        assert data.shape == labels.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TracrIITDataset(IITDataset):
    def __init__(self, base_data, ablation_data, hl_model, seed=0, every_combination=False):
        super().__init__(base_data, ablation_data, seed, every_combination)
        self.hl_model = hl_model
        self.device = hl_model.cfg.device

    @staticmethod
    def collate_fn(batch, hl_model, device=DEVICE):
        def get_encoded_input_from_torch_input(xy):
            """Encode input to the format expected by the model"""
            x, y = zip(*xy)
            encoded_x = hl_model.map_tracr_input_to_tl_input(x)
            
            if hl_model.is_categorical():
                y = list(y)
                for i in range(len(y)):
                    y[i] =[0] + hl_model.tracr_output_encoder.encode(y[i][1:])
                y = list(map(list, zip(*y)))
                y = torch.tensor(y, dtype=torch.long).transpose(0, 1)
                # print(y, y.shape)
                num_classes = len(hl_model.tracr_output_encoder.encoding_map.keys())
                y = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
            else:
                y = list(map(list, zip(*y)))
                y[0] = list(np.zeros(len(y[0])))
                y = torch.tensor(y, dtype=torch.float32).transpose(0, 1)
            intermediate_values = None
            return encoded_x.to(device), y.to(device), intermediate_values

        base_input, ablation_input = zip(*batch)
        encoded_base_input = get_encoded_input_from_torch_input(base_input)
        encoded_ablation_input = get_encoded_input_from_torch_input(ablation_input)
        return encoded_base_input, encoded_ablation_input

    def make_loader(
        self,
        batch_size,
        num_workers,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: self.collate_fn(x, self.hl_model, self.device),
        )


def create_dataset(case, hl_model, train_count=12000, test_count=3000):
    data = case.get_clean_data(count=train_count + test_count)
    inputs = data.get_inputs().to_numpy()
    outputs = data.get_correct_outputs().to_numpy()
    train_inputs = inputs[:train_count]
    test_inputs = inputs[train_count:]
    train_outputs = outputs[:train_count]
    test_outputs = outputs[train_count:]
    train_data = TracrDataset(train_inputs, train_outputs)
    test_data = TracrDataset(test_inputs, test_outputs)
    return TracrIITDataset(train_data, train_data, hl_model), TracrIITDataset(test_data, test_data, hl_model)
