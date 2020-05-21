import numpy as np
import pandas as pd





def _brownian_bridge(t0, t1, t, t0_val, t1_val, _norms):
    """
    Generating intermediate value for two side pinned Brownian motion.
    All inputs are expected to have the same dimension if type is array alike.  Scalers will be broadcasted.
    :param t0: start time (yr)
    :param t1: end time(yr)
    :param t:  timestamp at which value will be generated (yr)
    :param t0_val: Brownian motion value at time t0
    :param t1_val: Brownian motion value at time t0
    :param _norms: standard normal random number (from quasi RNG)
    :return: Brownian motion value at time t0
    """
    _full_time_range = t1 - t0
    _left_time_range = t-t0
    _right_time_range = t1 - t
    _mean = t0_val*_right_time_range/_full_time_range + t1_val*_left_time_range/_full_time_range # linear interpolated mean
    _var = _left_time_range*_right_time_range/_full_time_range
    return np.sqrt(_var)*_norms+_mean


def construct_bridge(time_vec,augmented_sobol_norm):
    # left first split:

    generated_index = [time_vec.index[0], time_vec.index[-1]]
    brownian_bridge_path = {str(time_vec.index[0]): np.sqrt(time_vec[time_vec.index[0] ])*augmented_sobol_norm[:,0],
                            str(time_vec.index[-1]): np.sqrt(time_vec[time_vec.index[-1] ])*augmented_sobol_norm[:,1]
                           }
    norm_idx = 2

    stopping_mark = False
    while(not stopping_mark):
        stopping_mark = True # if all number is filled , stopping mark will not be set to False and the program will be finished.
        for _i in range(len(generated_index)-1): # for each adjcent pair of index, find the (left) mid index
            _head, _tail = generated_index[_i], generated_index[_i+1]
            if _tail- _head < 2:
                continue
            _body =(_head+_tail)/2 if  (_head+_tail)%2==0 else (_head+_tail-1)/2 # left index has priority
            generated_index.insert(_i+1, _body)
            ############## generate middle point using brownian bridge method ################
            _middle_point_path = _brownian_bridge(t0 = time_vec[_head], t1 = time_vec[_tail],
                                            t = time_vec[_body], t0_val= brownian_bridge_path[str(_head)], t1_val=brownian_bridge_path[str(_tail)] , _norms= augmented_sobol_norm[:, norm_idx]
                                           )
            norm_idx = norm_idx+1
            brownian_bridge_path[str(_body)] = _middle_point_path
            stopping_mark = False
    generated_data = pd.DataFrame(brownian_bridge_path)
    generated_data.columns = np.array(generated_data.columns).astype(float).astype(int)

    generated_data = generated_data[sorted(generated_data.columns)]

    return generated_data.values




def _inverse_cholesky(rvs):
    """
    Perform inverse Cholesky transformation on high dimension standard normal variables.
    The input has shape n*T where n is the sample size, T is the dimension of variables.
    The transformed result has zero sample mean for each dimension t<T and identity covariance matrix across n samples.
    :param path: n*T ndarray
    :param dim:
    :return:
    """
    # todo:visualize the ma
    C_orig = pd.DataFrame(rvs).cov()
    L = np.linalg.cholesky(C_orig)
    transformed_rvs = np.dot(rvs, np.linalg.inv(L.T))
    return transformed_rvs