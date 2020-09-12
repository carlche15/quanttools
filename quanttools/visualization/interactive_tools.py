#
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
from matplotlib.widgets import Slider
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

class FunctionBody():
    # todo :abstract

    def __init__(self, func,pivot_x, base_y,init_param, cost_func = None, optimizer = None):
        #todo : identify length of parameter
        self.func = copy.copy(func)
        self.pivot_x, self.base_y = pivot_x, base_y
        self.init_param = init_param
        self.cost_func = cost_func if cost_func is not None else FittingEngine.rmse_cost
        self.optimizer = optimizer if optimizer is not None else FittingEngine.optimize
        self.calibrated_param = self.fit(self.pivot_x,self.base_y)

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
        pivot_line, = ax.plot(self.pivot_x, self.func(param,self.pivot_x), "-o", color="royalblue", markersize=25, lw=0)
        plotline, = ax.plot(domain, self.func(param, domain), lw=4, color="royalblue")
        return  pivot_line,plotline


class LineFitter():

    def __init__(self,func, pivotline, plotline, task_handler = None):
        # TODO: Keep this class as seperate as possible from function body class!
        """
        By default, when the mouse is released, the task_handler.run() will be called
        """
        self.func = func
        self.pivot_line = pivotline
        self.plotline = plotline
        self.pivot_xdata =pivotline.get_xdata() # a very good example for seperation
        self.pivot_ydata= pivotline.get_ydata()
        self.trackingidx = None # None or the idx that is being tracking

        self.text_var = self.plotline.figure.axes[0].text(1.6, 3, "Real time parameter:", fontsize=20) # todo: tidy this
        self.task_handler = task_handler

    def connect(self):

        self.cid_click = self.pivot_line.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_movie = self.pivot_line.figure.canvas.mpl_connect('motion_notify_event', self.tracking_mouse)
        self.cid_release = self.pivot_line.figure.canvas.mpl_connect('button_release_event', self.release_click)


    def on_click(self, event, eps = 1): # precision default to
        if event.inaxes is None:return # outside the box

        dist = np.hypot(self.pivot_line.get_xdata() - event.xdata, self.pivot_line.get_ydata() - event.ydata)
        if np.amin(dist) < eps:

            self.trackingidx = np.argmin(dist)
        else: return

    def tracking_mouse(self,event):


        if self.trackingidx is None: return
        if event.inaxes is None: return  # outside the box
        self.pivot_ydata[self.trackingidx] = event.ydata

        #######
        self.func.fit(self.pivot_xdata, self.pivot_ydata)
        updated_plot_yvals = self.func(self.plotline.get_xdata())


        ####update plot line
        self.plotline.set_ydata(updated_plot_yvals)
        self.plotline.figure.canvas.draw()

        #### update pivot line
        self.pivot_line.set_ydata(self.pivot_ydata)
        self.text_var.set_text(f"Real time parameter: {self.func.calibrated_param}")
        self.pivot_line.figure.canvas.draw()


    def release_click(self,event):

        if self.task_handler is not None and self.trackingidx is not None:
            print(f"Re-computing based on index movement {self.trackingidx}....")
            self.task_handler.run(self.func.calibrated_param)

        self.trackingidx = None

