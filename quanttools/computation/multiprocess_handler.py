from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import quanttools.computation._data_storage as _ds
import numpy as np
import time
import ctypes

def initProcess(share):
  _ds.toShare = share

def single_run(idx,model,params):

    mp_arr = _ds.toShare
    return model(params,np.frombuffer(mp_arr.get_obj()).reshape([8, 30, 18, 20])[None,idx-1,:])

class MpHandler():
    def __init__(self,data, func):
        print("____initializing multiprocessing handler______")
        mp_arr = mp.Array(ctypes.c_double, data.ravel())  # shared, can be used from multiple processes
        self.task_function = func # todo: handle multi function

        self.pools = Pool(40, initializer=initProcess, initargs=(mp_arr,))
        print("_______ MPH initialized! :D ______")
    def run(self, params):
        st = time.time()
        self.idxs = [(i, self.task_function, params, ) for i in range(8)]
        self.pools.starmap(single_run, self.idxs)
        print(time.time()-st)