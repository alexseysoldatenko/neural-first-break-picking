import numpy as np
import pandas as pd
import torch
import segyio
import napari
import os
import matplotlib.pyplot as plt

class Pick():
    def __init__(self, dt, trace_numbers, pick_time, shift = 0, time_shift = 0):
        self.dt = dt
        self.trace_numbers = trace_numbers - trace_numbers[0] - shift
        self.pick_time = pick_time - time_shift
    
    def get_pick_as_points_to_napari(self):
        return np.concatenate([self.pick_time.reshape(-1,1), self.trace_numbers.reshape(-1,1)], axis = 1)

def read_pick_excel(excel_path,**qwargs):
    raw_data = pd.read_excel(excel_path)
    trace_number = np.array(raw_data['TRACENO'])
    pick_time = np.array(raw_data['FBPICK']/raw_data['dt'], dtype=np.int32)

    dt = raw_data['dt'].iloc[0]

    if 'time_shift' in qwargs:
        return Pick(dt, trace_number, pick_time, time_shift=qwargs['time_shift'])
    else:
        return Pick(dt, trace_number, pick_time)


def read_pick_kingdom(path_to_pick, dt = 0.000125,**qwargs):
    with open(path_to_pick, 'r') as f:
        all_traces = f.read().split('\n')[:-1]
        trace_number = []
        pick_time = []
        for trace in all_traces:
            split_trace = trace.strip().split(' ')
            if len(split_trace) == 5:
                trace_number.append(int(float(split_trace[1])))
                pick_time.append(float(split_trace[4]))
            else:
                trace_number.append(int(float(split_trace[1])))
                pick_time.append(float(split_trace[2]))
    if 'time_shift' in qwargs:
        return Pick(dt, np.array(trace_number), (np.array(pick_time)/ dt).astype(np.int32), time_shift=qwargs['time_shift'])
    else:
        return Pick(dt, np.array(trace_number), (np.array(pick_time)/ dt).astype(np.int32))
    
def from_data_pick_to_torch_tensor(number_of_images, image_size, data, pick: Pick, bin_size = 64, 
                                   x_folder = 'x',y_folder='y',prefix=''):
    """
        Work only if pick in right format ¯\_(ツ)_/¯
    """
    if not os.path.exists(x_folder):
            os.makedirs(x_folder)
    if not os.path.exists(y_folder):
            os.makedirs(y_folder)
    for num in range(number_of_images):
        left_trace_num = np.random.randint(0,data.shape[1]-image_size)
        sub_image = data[:,left_trace_num:left_trace_num + image_size]
        sub_pick = pick.pick_time[left_trace_num:left_trace_num + image_size]
        zero_padding = np.zeros((int(sub_image.shape[0]//bin_size * (bin_size + 1) - sub_image.shape[0]), sub_image.shape[1]))
        sub_image = np.concatenate([sub_image,zero_padding])
        mask_image = np.zeros_like(sub_image)
        for trace_num in range(mask_image.shape[1]):
            mask_image[:sub_pick[trace_num],trace_num] = 0
            mask_image[sub_pick[trace_num]:,trace_num] = 1
        sub_image_path = f"{x_folder}\\{prefix}_{num}.pt"
        mask_image_path = f"{y_folder}\\{prefix}_{num}.pt"
        torch.save(torch.from_numpy(sub_image), sub_image_path)
        torch.save(torch.from_numpy(mask_image), mask_image_path)
        print(f'{num}_ready')
    
def load_data(path_to_data):
    with segyio.open(path_to_data,  ignore_geometry = True) as f:
        data = f.trace.raw[:].T
    return data

def view_data(path_to_data, pick, dt = 0.000125,**qwargs):
    data = load_data(path_to_data)
    if 'kingdom' in qwargs:
        pick = read_pick_kingdom(pick, dt = dt, **qwargs).get_pick_as_points_to_napari()
    if 'excel' in qwargs:
        pick = read_pick_excel(pick, **qwargs).get_pick_as_points_to_napari()
    
    viewer = napari.view_image(data)
    viewer.add_points(pick,edge_color = 'red', size = 3,face_color = 'red')
    napari.run()

if __name__ == '__main__':
   
    path_to_pick = 'C:\\Users\\alexsey\\Desktop\\проекты\\пикировка первых вступлений грант\\пикировки\\100.xlsx'
    path_to_data = 'C:\\Users\\alexsey\\Desktop\\проекты\\пикировка первых вступлений грант\\данные\\WSBS22_3D_line000_geom_sou1rec2.sgy'
    view_data(path_to_data, path_to_pick, dt = 1, excel = True)#,time_shift=70




    # data = load_data(path_to_data)
    # x_folder = 'C:\\Users\\alexsey\\Desktop\\x'
    # y_folder = 'C:\\Users\\alexsey\\Desktop\\y'
    # pick = read_pick_kingdom(path_to_pick, dt = 0.0005,time_shift = 5)
    # for i in [64,32,16,128,256]:
    #     from_data_pick_to_torch_tensor(100, i, data, pick, x_folder=x_folder, y_folder=y_folder, prefix =f"Line3_{i}")