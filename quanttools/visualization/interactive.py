#
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
from matplotlib.widgets import Slider
from quanttools.visualization.interactive_tools import  *


##### defines the functions to be plotted
burnout_inputs = (lambda parameter, xs:  np.arctan(parameter[0] * (np.array(xs) + parameter[1])) + np.pi / 2, [-5,-1],
                  FittingEngine.rmse_cost, FittingEngine.optimize)



if __name__ == "__main__":


    epsilon = 50
    xs =[0,0.5,1,2,3]
    ys = [3,2.75,1.55,0.2,0.1]
    fig,ax1 = plt.subplots(figsize=(6,6))


    ###### base line function ###############
    # todo : move this inside
    burnout_function = FunctionBody(*burnout_inputs)
    burnout_function.fit(xs,ys)
    baseline, = ax1.plot(np.linspace(xs[0],xs[-1],100),burnout_function(np.linspace(xs[0],xs[-1],100)),'--',lw=4,color="gray")
    plotline, = ax1.plot(np.linspace(xs[0],xs[-1],100),burnout_function(np.linspace(xs[0],xs[-1],100)),lw=4,color="royalblue")
    pivot_line,= ax1.plot(xs,burnout_function(xs), "-o",color="royalblue", markersize=25,lw=0)


    #########plotting engine############
    lc = LineFitter(pivot_line,burnout_function,finer_plot=True, plotline=plotline)
    lc.connect()

    plt.show()