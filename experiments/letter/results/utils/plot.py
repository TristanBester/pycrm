import numpy as np
import pandas as pd


def _standardize_length(x_values, y_values, x_min, x_max, step):
    df_result = pd.DataFrame(
        {
            "x": x_values,
            "y": y_values,
        }
    )
    results = df_result.groupby("x").y.mean()
    x_values = np.array(results.index)
    y_values = np.array(results.values)

    xs = np.arange(x_min, x_max, step)
    ys = np.interp(xs, x_values, y_values)
    return ys
