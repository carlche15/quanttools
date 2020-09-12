
from quanttools.visualization.interactive_tools import  *
from quanttools.computation.multiprocess_handler import *

##### defines the functions to be plotted
def burnout(parameter, xs):
    return np.arctan(parameter[0] * (np.array(xs) + parameter[1])) + np.pi / 2 # todo: change cdf to arctan cauz it is faster!
def seasoning(parameter, xs):

    return 1-np.exp(-parameter[0]*np.array(xs))

# function itself, x, y (for baseline), init parameter, cost func. optimizer
burnout_inputs = (burnout, [0,0.5,1,2,3], [3,2.75,1.55,0.2,0.1],[-5,-1], None, None)
seasoning_inputs = (seasoning, [0,10,20,30,40,50], [0,0.2,0.6,0.8,0.9,0.95],[-5], None, None)



if __name__ == "__main__":


    # Create plots
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))


    ###### base line function ###############
    # todo : move this inside
    burnout_function = FunctionBody(*burnout_inputs)
    seasoning_function = FunctionBody(*seasoning_inputs)


    ######## data_handlers

    data = np.random.normal(size=[8, 30, 18, 20])
    mph = MpHandler(data, burnout_function.func)

    #########plotting engine############
    lc = LineFitter(burnout_function, *burnout_function.creat_2D_line(ax1),task_handler=mph)
    lc2 = LineFitter(seasoning_function, *seasoning_function.creat_2D_line(ax2),task_handler=None)
    lc.connect()
    lc2.connect()

    plt.show()