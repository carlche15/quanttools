import numpy as np
from scipy.optimize import fmin
import copy


class FittingEngine():

    @staticmethod
    def rmse_cost(parameters, func, xs, ys, **kwargs):
        line_fit = func(parameters, xs, **kwargs)
        true = ys
        return np.sqrt(np.mean((line_fit - true) ** 2))

    @staticmethod
    def optimize(init_pram,cost_func, func, xs, ys):
        res = fmin(cost_func, x0=init_pram, args=(func, xs, ys,), disp=False)
        return res

class FunctionMapping():
    # todo :abstract

    def __init__(self, func,pivot_x, base_y, init_param, cost_func = None, optimizer = None, id = None,calibrated_param = None):
        #todo : identify length of parameter
        self.func = copy.copy(func)
        self.init_param = init_param
        self.cost_func = cost_func if cost_func is not None else FittingEngine.rmse_cost
        self.optimizer = optimizer if optimizer is not None else FittingEngine.optimize
        self.id = id
        if calibrated_param is None:
            self.pivot_x, self.base_y = pivot_x, base_y  # todo: if calibrated parameters are providede, I should use them.
            self.calibrated_param = self.fit(self.pivot_x,self.base_y)
        else:
            self.calibrated_param = calibrated_param
            print(self.id,calibrated_param)
            self.pivot_x,self.base_y = pivot_x,self.func(self.calibrated_param,pivot_x)


    def __call__(self, xs):
        return self.func(self.calibrated_param,xs)

    def fit(self, xs,ys):
        self.calibrated_param = self.optimizer(self.init_param,self.cost_func,self.func,xs,ys)
        return self.calibrated_param

    def creat_2D_line(self, ax):
        """
        create 2 line plots on a given matplotlib axis and return the corresponding Lind2D object
        :param function: a function map x to y
        :param ax: axis on which lines are created
        :return: 0. （NOT RETURNED）the base-line for the function (gray-dash-line), this line is fixed all the time
                 1. Scatter ("bo" type line)  for x,y pairs, these points will become movable
                 2. Line plots for the function. This line will be updated based on the latest position of line 1

        """
        # todo: hard-coded 100, color,.etc
        domain = np.linspace(self.pivot_x[0], self.pivot_x[-1], 100)
        param = self.calibrated_param
        baseline, = ax.plot(domain, self.func(param,domain), '--', lw=4,color="gray")
        pivot_line, = ax.plot(self.pivot_x, self.func(param,self.pivot_x), "-o", color="royalblue", markersize=12, lw=0)
        plotline, = ax.plot(domain, self.func(param, domain), lw=4, color="royalblue")
        ax.set_title(f"{self.id}")
        return  pivot_line,plotline

    def update(self,plotter):
        self.fit(plotter.pivot_xdata,plotter.pivot_ydata)
        return plotter.pivot_ydata, self(np.linspace(self.pivot_x[0], self.pivot_x[-1], 100))


        #
        # ####update plot line
        # self.plotline.set_ydata(self(self.plotline.get_xdata()))
        # self.plotline.figure.canvas.draw()
        #
        # #### update pivot line
        # self.pivot_line.set_ydata(self.pivot_ydata)
        # # self.text_var.set_text(f"Real time parameter: {self.func.calibrated_param}")
        # self.pivot_line.figure.canvas.draw()
    #
    #
    #
    #
    #     #######
    #     self.func.fit(self.pivot_xdata, self.pivot_ydata)
    #     updated_plot_yvals = self.func(self.plotline.get_xdata())
    #
    #     ####update plot line
    #     self.plotline.set_ydata(updated_plot_yvals)
    #     self.plotline.figure.canvas.draw()
    #
    #     #### update pivot line
    #     self.pivot_line.set_ydata(self.pivot_ydata)
    #     # self.text_var.set_text(f"Real time parameter: {self.func.calibrated_param}")
    #     self.pivot_line.figure.canvas.draw()

