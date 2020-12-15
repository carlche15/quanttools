
import numpy as np
from scipy.optimize import fmin
import copy

class LineFitter():

    def __init__(self, pivotline, plotline,
                 click_handler = None, click_task = None,
                 move_handler = None, move_task = None,
                 release_handler = None, release_task = None):
        # TODO: Keep this class as seperate as possible from function body class!
        """
        When the mouse is released, the task_handler.run() will be called
        """
        self.pivot_line = pivotline
        self.plotline = plotline
        self.pivot_xdata =pivotline.get_xdata() # a very good example for seperation
        self.pivot_ydata= pivotline.get_ydata()
        self.trackingidx = None # None or the idx that is being tracking

        # self.text_var = self.plotline.figure.axes[0].text(1.6, 3, "Real time parameter:", fontsize=20) # todo: tidy this
        self.click_handler = click_handler
        self.click_task = click_task
        self.move_handler = move_handler
        self.move_task = move_task
        self.release_handler = release_handler
        self.release_task = release_task

    def connect(self):

        self.cid_click = self.pivot_line.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_movie = self.pivot_line.figure.canvas.mpl_connect('motion_notify_event', self.tracking_mouse)
        self.cid_release = self.pivot_line.figure.canvas.mpl_connect('button_release_event', self.release_click)

    def _run_task(self, handler, task):
        if task is None:
            return
        else:
            if handler is not None:
                return handler(task,self)
            else:

                return task(self)


    def on_click(self, event): # precision default to
        if event.inaxes is None:return # outside the box
        dist = np.hypot(self.pivot_line.get_xdata() - event.xdata, self.pivot_line.get_ydata() - event.ydata)
        eps = np.mean(np.diff(self.pivot_line.get_xdata()))/5  # click precision set to be 1/5 of average distance between pivot x points.
        if np.amin(dist) < eps:

            self.trackingidx = np.argmin(dist)
        else: return
        self._run_task(self.click_handler, self.click_task)

    def tracking_mouse(self,event):


        if self.trackingidx is None: return
        if event.inaxes is None: return  # outside the box
        self.pivot_ydata[self.trackingidx] = event.ydata

        # self.run_task(self.move_handler,self.move_task)

        #######
        pivot_y, plot_y = self._run_task(self.move_handler, self.move_task)
        self.plotline.set_ydata(plot_y)
        self.pivot_line.set_ydata(pivot_y)
        # self.text_var.set_text(f"Real time parameter: {self.func.calibrated_param}")

        self.plotline.figure.canvas.draw()
        self.pivot_line.figure.canvas.draw()




    def release_click(self,event):

        if self.trackingidx is not None:
            print(f"Re-computing based on index movement {self.trackingidx}....")

            self._run_task(self.release_handler, self.release_task)

        self.trackingidx = None



