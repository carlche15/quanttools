#
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
from matplotlib.widgets import Slider

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

    def __init__(self, init_param, cost_func, optimizer):
        self.init_param = init_param
        self.cost_func = cost_func
        self.optimizer = optimizer
        self.calibrated_param = None

    def __call__(self, xs):
        return self.func(self.calibrated_param,xs)

    def func(self, parameter, xs):
        m = parameter[0]
        d = parameter[1]
        return np.arctan(m * (np.array(xs) + d)) + np.pi / 2

    def fit(self, xs,ys):
        self.calibrated_param = self.optimizer(self.init_param,self.cost_func,self.func,xs,ys)
        return self.calibrated_param

class LineFitter():

    def __init__(self, pivotline, func, finer_plot=True, plotline = None):
        self.pivot_line = pivotline
        self.func = func
        self.pivot_xdata =pivotline.get_xdata()
        self.pivot_ydata= pivotline.get_ydata()
        self.trackingidx = None # None or the idx that is being tracking
        self.finer_plot = finer_plot
        self.plotline = plotline
        self.text_var = self.plotline.figure.axes[0].text(1.6, 3, "Real time parameter:", fontsize=20)

    def connect(self):

        self.cid_click = self.pivot_line.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_movie = self.pivot_line.figure.canvas.mpl_connect('motion_notify_event', self.tracking_mouse)
        self.cid_release = self.pivot_line.figure.canvas.mpl_connect('button_release_event', self.release_click)


    def on_click(self, event, eps = 0.1): # precision default to
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
        self.trackingidx = None








epsilon = 50
xs =[0,0.5,1,2,3]
ys = [3,2.75,1.55,0.2,0.1]
init_param = [-5,-1]
fig,ax1 = plt.subplots(figsize=(20,10))


###### base line function ###############
# todo : move this inside
burnout_function = FunctionBody(init_param,cost_func=FittingEngine.rmse_cost, optimizer=FittingEngine.optimize)
burnout_function.fit(xs,ys)
baseline, = ax1.plot(np.linspace(xs[0],xs[-1],100),burnout_function(np.linspace(xs[0],xs[-1],100)),'--',lw=4,color="gray")
plotline, = ax1.plot(np.linspace(xs[0],xs[-1],100),burnout_function(np.linspace(xs[0],xs[-1],100)),lw=4,color="royalblue")
pivot_line,= ax1.plot(xs,burnout_function(xs), "-o",color="royalblue", markersize=25,lw=0)


#########plotting engine############
lc = LineFitter(pivot_line,burnout_function,finer_plot=True, plotline=plotline)
lc.connect()

plt.show()