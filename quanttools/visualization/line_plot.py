import pandas as pd
import numpy as np
from quanttools.statistics.data_process import winsorize

# todo: change font


def two_axis_plot(_data, d1, d2, target_var_1, target_var_2, weights = None, title=None, plot_diff=False, **kwargs):
    """
    :param _data:
    :param d1: bucketed varaible
    :param d2: x-axis variable
    :param target_var_1:
    :param weights:
    :return: None
    """
    from decimal import Decimal
    import matplotlib.pyplot as plt
    data=_data.copy()
    #todo: performance

    data["_outer_bucket"]=[i.mid for i in pd.qcut(data[d1],6)]
    color_dir=["lightblue","skyblue","deepskyblue","dodgerblue","royalblue","mediumblue"]
    fig, ax = plt.subplots(figsize=kwargs.get('figsize',(25,9)))

    for idx, ob in enumerate(np.unique(data["_outer_bucket"])):
        _sub_data=data.query(f"_outer_bucket=={ob}")
        _line_1=one_axis_avg(_sub_data, bucket_var=d2, target_var = target_var_1, weights=weights)
        _line_2=one_axis_avg(_sub_data, bucket_var=d2, target_var=target_var_2, weights=weights)
        if plot_diff:
            ax.plot(_line_1-_line_2, color=color_dir[idx],lw=kwargs.get("lw",4),label='%.2E' % Decimal(ob))
        else:
            ax.plot(_line_1,color=color_dir[idx],lw=kwargs.get("lw",4),label='%.2E' % Decimal(ob))
            ax.plot(_line_2,"--",color=color_dir[idx],lw=kwargs.get("lw",4))

    if title is None:
        ax.set_title(f"{d1} and {d2} on {target_var_1} / {target_var_2}", fontsize=kwargs.get('fontsize',24))
    else:
        ax.set_title(title, fontsize=kwargs.get('fontsize',24))
    ax.legend(title=f"{d1} buckets \n-- for {target_var_2}\n - for {target_var_1}",fontsize=kwargs.get('fontsize',24),title_fontsize=kwargs.get('fontsize',24))
    ax.set_xlabel(d2,fontsize=kwargs.get('fontsize',24))
    ax.tick_params(axis='both', which='major', labelsize=kwargs.get('labelsize',20))
    ax.set_ylabel(f"{target_var_1}/{target_var_2}",fontsize=kwargs.get('fontsize',24))
    plt.show()
    return fig


def one_axis_avg(data, bucket_var, target_var, weights=None):

    """
    compute the average of the target variable along the bucket variable.

    """
    distinct_bucket_interval = len(np.unique(data[bucket_var]))  # distinct number of x-axis variable

    # if the number of distinct feature values is greater than 100, create 20 even cut buckets for it.
    # (because it would be undesirable to grouby by so many values)
    # if the number of distinct feature values is less than 100, group directly by feature values
    if distinct_bucket_interval > 100:

        data = winsorize(data, **{bucket_var:(1,99)})
        bucket = data.groupby( pd.cut(data[bucket_var], 30)).apply(lambda x: np.average(x[target_var], weights=x[weights] if weights else np.ones(len(x))) if len(x)>50 else np.nan)
        bucket.index = [i.mid for i in bucket.index]
    else:
        bucket = data.groupby([bucket_var]).apply(lambda x: np.average(x[target_var], weights=x[weights] if weights else np.ones(len(x)) ))

    return bucket
