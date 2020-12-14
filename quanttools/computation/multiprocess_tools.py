from multiprocessing import Pool
import multiprocessing as mp
import quanttools.computation._data_storage as _ds
import numpy as np
import time
import ctypes


def init_process(share):
    _ds.toShare = share


def single_run(idx, model):
    """
    :param idx:
    :param model:
    :return:
    """
    mp_arr = _ds.toShare
    # return params(np.frombuffer(mp_arr.get_obj()).reshape([8, 400, 180, 20])[None, idx - 1, :])
    return model(np.frombuffer(mp_arr.get_obj()).reshape([8, 30, 36, 7] )[None, idx-1, :])


class MpHandler:
    def __init__(self, data):
        print("____initializing multiprocessing handler______")
        mp_arr = mp.Array(ctypes.c_double, data.ravel())  # shared, can be used from multiple processes
        self.pools = Pool(8, initializer=init_process, initargs=(mp_arr,))
        print("_______ MPH initialized! :D ______")

    def run(self,func):
        st = time.time()
        idx = [(i, func, ) for i in range(8)]
        self.pools.starmap(single_run, idx)
        print(time.time()-st)
