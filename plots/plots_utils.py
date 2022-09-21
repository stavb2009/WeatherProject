import os

import numpy as np
import pandas as pd
import plotly as px
import plotly.express as ppx

def import_csv_to_pandas(dir):
    try:
        csv_file = pd.read_csv(dir)
        return csv_file
    except:
        print(f"def import_csv_to_pandas: couldn't read the csv file with dir: {dir}")


def from_XY_to_RTheta(x, y) -> tuple:
    x, y = float(x), float(y)
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan2(y, x))
    return (r, theta)


def plot_rose_diagram(output: tuple, forecast: tuple):
    '''
    The inputs and forecast are in (X, Y) format
    :param output:
    :param forecast:
    :return:
    '''

    output_r, output_theta = from_XY_to_RTheta(*output)
    forecast_r, forecast_theta = from_XY_to_RTheta(*forecast)

    data = {'names': ['output', 'forecast'], 'radius': [output_r, forecast_r], 'theta': [output_theta, forecast_theta]}
    df = pd.DataFrame(data=data)

    fig = ppx.scatter_polar(df, r="radius", theta="theta",hover_name='names', color="names",template="plotly_dark")
    fig.show()


def plot_those_indexes(results, forecast, raw, colums):
    results_X = results.iloc[raw][colums]
    results_Y = results.iloc[raw+1][colums]
    forecast_X = forecast.iloc[raw+1][colums]
    forecast_Y = forecast.iloc[raw+2][colums]
    plot_rose_diagram((results_X, results_Y), (forecast_X, forecast_Y))




if __name__ == '__main__':
    output_dir = 'test_output_results.csv'
    results = import_csv_to_pandas(output_dir)
    results_X = results.iloc[3][20]
    results_Y = results.iloc[4][20]

    forecast_dir = 'test_forecast_new_pos.csv'
    forecast = import_csv_to_pandas(forecast_dir)
    forecast_X = forecast.iloc[4][20]
    forecast_Y = forecast.iloc[5][20]

    plot_rose_diagram((results_X, results_Y), (forecast_X, forecast_Y))

    plot_those_indexes(results, forecast, 3, 35)
    plot_those_indexes(results, forecast, 3, 40)

    plot_those_indexes(results, forecast, 35, 12)
    plot_those_indexes(results, forecast, 35, 17)

    plot_those_indexes(results, forecast, 39, 50)
    plot_those_indexes(results, forecast, 39, 50)
    print('check')
