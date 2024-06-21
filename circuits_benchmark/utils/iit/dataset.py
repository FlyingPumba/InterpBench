import numpy as np
from torch.utils.data import Dataset, DataLoader
from iit.utils.config import DEVICE
import torch
from iit.utils.iit_dataset import IITDataset
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase


# class TracrIITDataset(IITDataset):
#     def __init__(self, base_data, ablation_data, hl_model, seed=0, every_combination=False):
#         super().__init__(base_data, ablation_data, seed, every_combination)
#         self.hl_model = hl_model
#         self.device = hl_model.device
#
#     @staticmethod
#     def collate_fn(batch, hl_model, device=DEVICE):
#         base_input, ablation_input = zip(*batch)
#         encoded_base_input = get_encoded_input_from_torch_input(base_input, hl_model, device)
#         encoded_ablation_input = get_encoded_input_from_torch_input(ablation_input, hl_model, device)
#         return encoded_base_input, encoded_ablation_input
#
#     def make_loader(
#         self,
#         batch_size,
#         num_workers,
#     ) -> DataLoader:
#         return DataLoader(
#             self,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=num_workers,
#             collate_fn=lambda x: self.collate_fn(x, self.hl_model, self.device),
#         )


# class TracrUniqueDataset(TracrIITDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __getitem__(self, index):
#         return self.base_data[index]
#
#     def __len__(self):
#         return len(self.base_data)
#
#     @staticmethod
#     def collate_fn(batch, hl_model, device=DEVICE):
#         encoded_base_input = get_encoded_input_from_torch_input(batch, hl_model, device)
#         return encoded_base_input


# def create_dataset(case: BenchmarkCase, hl_model, test_frac=0.2, min_train_count=20000, max_train_count=120_000):
#     # total_len = get_total_len(case)
#     # # get all data if in the range of min_train_count and max_train_count
#     # count = int(min_train_count * (1 + test_frac)) if total_len < min_train_count \
#     #     else total_len if total_len < max_train_count \
#     #     else int(max_train_count * (1 + test_frac))
#     # train_count = int(count * (1 - test_frac))
#     # data = case.get_clean_data(count=count)
#     #
#     # inputs = data.get_inputs().to_numpy()
#     # outputs = data.get_correct_outputs().to_numpy()
#     #
#     # train_inputs = inputs[:train_count]
#     # test_inputs = inputs[train_count:]
#     # train_outputs = outputs[:train_count]
#     # test_outputs = outputs[train_count:]
#     train_data = TracrDataset(train_inputs, train_outputs)
#     test_data = TracrDataset(test_inputs, test_outputs)
#
#     return TracrIITDataset(train_data, train_data, hl_model), TracrIITDataset(test_data, test_data, hl_model)


# def get_unique_data(case: BenchmarkCase, max_len=10_000):
#     '''
#     Returns all possible unique datapoints from the case
#     if the number of unique datapoints is less than max_len.
#     Otherwise, returns a random sample of max_len unique datapoints.
#     '''
#     total_len = get_total_len(case)
#
#     if total_len < max_len:
#         data = case.get_clean_data(count=total_len)
#         test_inputs = data.get_inputs().to_numpy()
#         test_outputs = data.get_correct_outputs().to_numpy()
#         unique_test_data = TracrDataset(test_inputs, test_outputs)
#         return unique_test_data
#
#     data = case.get_clean_data(count=3*max_len)
#     test_inputs = data.get_inputs().to_numpy()
#     test_outputs = data.get_correct_outputs().to_numpy()
#     arr, idxs = np.unique([", ".join(str(i)) for i in np.array(test_inputs)], return_inverse=True)
#     # create indices that point to the first unique input
#     all_possible_inputs = np.arange(arr.shape[0])
#     # find the first occurence of all_possible_inputs in idxs
#     first_occurences = [np.where(idxs == i)[0][0] for i in all_possible_inputs]
#
#     unique_test_inputs = test_inputs[first_occurences]
#     unique_test_outputs = test_outputs[first_occurences]
#     # assert len(unique_test_inputs) == len(unique_test_outputs)
#     # assert len(unique_test_inputs) == len(np.unique([", ".join(str(i)) for i in np.array(test_inputs)]))
#     # assert len(np.unique([", ".join(str(i)) for i in np.array(unique_test_inputs)])) == len(unique_test_inputs)
#     if len(unique_test_inputs) > max_len:
#         random_indices = np.random.choice(len(unique_test_inputs), max_len, replace=False)
#         unique_test_inputs = unique_test_inputs[random_indices]
#         unique_test_outputs = unique_test_outputs[random_indices]
#     unique_test_data = TracrDataset(unique_test_inputs, unique_test_outputs)
#     return unique_test_data
