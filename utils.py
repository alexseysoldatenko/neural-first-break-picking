import numpy as np
import torch
import segyio
import napari

class Pick():
    def __init__(self, dt, trace_numbers, pick_time, shift = 0, time_shift = 0):
        self.dt = dt
        self.trace_numbers = trace_numbers - shift
        self.pick_time = pick_time - time_shift
    
    def get_pisk_as_points_to_napari(self):
        return np.concatenate([self.pick_time.reshape(-1,1), self.trace_numbers.reshape(-1,1)], axis = 1)



def read_pick_kingdom(path_to_pick, dt = 0.000125):
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
    return Pick(dt, np.array(trace_number), (np.array(pick_time)/ dt).astype(np.int32))
    
def from_data_pick_to_torch_tensor(data, pick, save_folder = 'test\\'):
    """
        Work only if pick in right format ¯\_(ツ)_/¯
    """
    

def view_data(path_to_data, pick, dt = 0.000125):
    with segyio.open(path_to_data,  ignore_geometry = True) as f:
        data = f.trace.raw[:].T
    pick = read_pick_kingdom(pick, dt = dt).get_pisk_as_points_to_napari()
    
    viewer = napari.view_image(data)
    viewer.add_points(pick,edge_color = 'red', size = 3,face_color = 'red')
    napari.run()

if __name__ == '__main__':
   
    path_to_pick = 'C:\\Users\\alexsey\\Desktop\\проекты\\пикировка первых вступлений грант\\пикировки\\Line5.dat'
    path_to_data = 'C:\\Users\\alexsey\\Desktop\\проекты\\пикировка первых вступлений грант\\данные\\Line5.sgy'
    view_data(path_to_data, path_to_pick, dt = 0.00003657)
    # read_pick_kingdom(path_to_pick)