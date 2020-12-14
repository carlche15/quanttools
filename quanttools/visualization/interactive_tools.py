
import numpy as np
from scipy.optimize import fmin
import copy

class LineFitter():

    def __init__(self,func, pivotline, plotline, task_handler = None, task = None):
        # TODO: Keep this class as seperate as possible from function body class!
        """
        When the mouse is released, the task_handler.run() will be called
        """
        self.func = func
        self.pivot_line = pivotline
        self.plotline = plotline
        self.pivot_xdata =pivotline.get_xdata() # a very good example for seperation
        self.pivot_ydata= pivotline.get_ydata()
        self.trackingidx = None # None or the idx that is being tracking

        # self.text_var = self.plotline.figure.axes[0].text(1.6, 3, "Real time parameter:", fontsize=20) # todo: tidy this
        self.task_handler = task_handler
        self.task = task if task is not None else self.func

    def connect(self):

        self.cid_click = self.pivot_line.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_movie = self.pivot_line.figure.canvas.mpl_connect('motion_notify_event', self.tracking_mouse)
        self.cid_release = self.pivot_line.figure.canvas.mpl_connect('button_release_event', self.release_click)


    def on_click(self, event): # precision default to
        if event.inaxes is None:return # outside the box
        dist = np.hypot(self.pivot_line.get_xdata() - event.xdata, self.pivot_line.get_ydata() - event.ydata)
        eps = np.mean(np.diff(self.pivot_line.get_xdata()))/5  # click precision set to be 1/5 of average distance between pivot x points.
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
        # self.text_var.set_text(f"Real time parameter: {self.func.calibrated_param}")
        self.pivot_line.figure.canvas.draw()


    def release_click(self,event):


        if self.task_handler is not None and self.task is not None and self.trackingidx is not None:
            print(f"Re-computing based on index movement {self.trackingidx}....")
            # func satisfies f(x)->y
            print(self.func.calibrated_param)
            self.task_handler.run(self.task)

        self.trackingidx = None



