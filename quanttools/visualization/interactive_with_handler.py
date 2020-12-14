
from quanttools.visualization.interactive_tools import  *
from quanttools.visualization.function_tools import  *
from quanttools.computation.multiprocess_tools import *
import matplotlib.pyplot as plt
##### defines the functions to be plotted
def burnout(parameter, xs):
    return np.arctan(parameter[0] * (np.array(xs) + parameter[1])) + np.pi / 2 # todo: change cdf to arctan cauz it is faster!
def seasoning(parameter, xs):
    return 1-np.exp(-parameter[0]*np.array(xs))

class Model:
    def __init__(self, burnout_function,seasoning_function):
        self.burnout_function = burnout_function
        self.seasoning_function = seasoning_function
        self.key2index = {'refi_incentive': 0, 'burnout': 1, 'seasoning': 2,
                          'CURR_LOANS_SIZE': 3, 'lockin_incentive': 4, 'eff_mon': 5}

    def __call__(self,xs):
        # todo: allocate xs, this allocation should be consistent with variable generation methods.
        # todo: formalize id

        return 1/100*self.burnout_function(xs[:,:,:,self.key2index[self.burnout_function.id]])*\
               self.seasoning_function(xs[:,:,:,self.key2index[self.seasoning_function.id]])

# function itself, x, y (for baseline), init parameter, cost func. optimizer
burnout_inputs = (burnout, [0,0.5,1,2,3], [3,2.75,1.55,0.2,0.1],[-5,-1], None, None,"burnout")
seasoning_inputs = (seasoning, [0,10,20,30,40,50], [0,0.2,0.6,0.8,0.9,0.95],[-5], None, None,"seasoning")




if __name__ == "__main__":


    # Create plots
    fig,axes = plt.subplots(2,2,figsize=(12,6))
    (ax1,ax2,ax3,ax4) = axes.ravel()

    # Create Computation handlers; Overhead including written data

    data = np.load("../../inputs.npy",allow_pickle=True)
    mph = MpHandler(data)


    ###### base line function ###############
    # todo : move this inside
    burnout_function = FunctionMapping(*burnout_inputs)
    seasoning_function = FunctionMapping(*seasoning_inputs)
    model = Model(burnout_function,seasoning_function)


    #########plotting engine############
    lc = LineFitter(model.burnout_function, *model.burnout_function.creat_2D_line(ax1),task_handler=mph,task = model)
    lc2 = LineFitter(model.seasoning_function, *model.seasoning_function.creat_2D_line(ax2),task_handler=mph, task = model)
    lc.connect()
    lc2.connect()

    plt.show()