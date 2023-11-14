import segyio
import numpy as np
import pandas as pd
import torch
import json


class DatasetCreator:
    def __init__(self, main_config_path, data_config_path):
        with open(main_config_path, encoding='utf-8') as config:
            self.main_config = json.load(config)
        self.data_config = pd.read_excel(data_config_path)
        self.save_image_index = 0

    def prepare_picks(self):
        self.picks = []
        for index, row in self.data_config.iterrows():
            if row[0]['source'] == 'kingdom':
                self.picks.append(self.read_kingdom_pick(row[0]['path'], 
                                                         row[0]['dt'], 
                                                         row[0]['left_shift']))
            elif row[0]['source'] == 'radex':
                self.picks.append(self.read_radex_pick(row[0]['path'], 
                                                       row[0]['dt'], 
                                                       row[0]['left_shift']))

    def create_dataset(self):
        for index, row in self.data_config.iterrows():
            pick = self.picks[index]
            data_sgy = DatasetCreator.read_segy(row[0]['path_to_sgy'])
            self.get_data(data_sgy, pick, index)
                   
    def read_kingdom_pick(path : str, dt : float, left_shift : int) -> pd.DataFrame:
        pd.read_csv(path, sep=' ', header=None)
        data = data[[5]].rename(columns={ 5:'time'})
        data['time'] = data['time'].astype('float')
        data['time'] = np.round(data['time'] / dt).astype('Int64')
        return np.array(data)
    
    def read_radex_pick(path : str, dt : float, left_shift : int) -> pd.DataFrame:
        pass
    
    def read_segy(path: str, left_shift : int) -> np.ndarray: 
        with segyio.open(path,  ignore_geometry = True) as f:
            data = f.trace.raw[:].T
        data = data[:, left_shift:]
        return data
    
    def get_data(self, data : np.ndarray, pick : np.ndarray, index_config : int) -> None:
        assert data.shape[1] == pick.shape[0], "Data and pick must have same length"
        layers = self.data_config['layers'][index_config]
        height = self.data_config['height'][index_config]
        width = self.data_config['width'][index_config]
        step_width = self.data_config['step_width'][index_config]
        for j in range(layers):
            for i in range(int(data.shape[1]/step_width)-1):
                sub_data = data[height*j:height*(j+1), step_width*i:step_width*i + width]
                sub_pick = pick[step_width*i:step_width*i + width]
                image_path = self.main_config['save_folder'] + '/data/' + str(self.save_image_index)
                mask_path = self.main_config['save_folder'] + '/mask/' + str(self.save_image_index)
                DatasetCreator.create_image(sub_data, image_path, self.save_image_index)
                DatasetCreator.create_mask(sub_data.shape, sub_pick, mask_path, self.save_image_index)
                self.save_image_index += 1
        
    def create_image(data : np.ndarray, path : str, index : int) -> None:
        data = np.reshape(data, (1,data.shape[0], data.shape[1]))
        data = DatasetCreator.standardize(data)
        torch.save(torch.tensor(data), path + '/' + str(index) + '.pt')
    
    def create_mask(data_shape : tuple, pick : np.ndarray, path : str, index : int) -> None:
        mask = np.zeros(data_shape)
        for i in range(pick.shape[0]):
            mask[i, int(pick[i])] = 1
        torch.save(torch.tensor(mask), path + '/' + str(index) + '.pt')
        
    def standardize(data : np.ndarray) -> np.ndarray:
        data = (data - np.mean(data)) / np.std(data)
        return data
    







if __name__ == "__main__":
    path = r"C:\Users\alexsey\Desktop\проекты\тест данных от Ксюши для пикировки\files_to_mask.xlsx"
    main_config = "main_config.json"
    DatasetCreator(main_config,path).create_dataset()