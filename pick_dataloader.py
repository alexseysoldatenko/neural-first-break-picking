import torch
import os
import random




class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder_path):
        'Initialization'
        self.data_folder_path = data_folder_path
        self.dict_images = {}
        self.list_names = []
        for file in os.listdir(self.data_folder_path):
            filename = os.fsdecode(file)
            prefix = filename[:filename.index('_')]
            if prefix in self.dict_images:
                self.dict_images[prefix] += 1
            else:
                self.dict_images[prefix] = 1
            self.list_names.append(filename)
        random.shuffle(self.list_names)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        filename = self.list_names[index]
        # Load data and get label
        X = torch.load(f'{self.data_folder_path}/{filename}')[None,:,:].float()

        return X