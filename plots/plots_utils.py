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

    fig = ppx.bar_polar(df, r="radius", theta="theta",hover_name='names' ,template="plotly_dark")
    fig.show()







if __name__ == '__main__':
    output_dir = 'output_results.csv'
    results = import_csv_to_pandas(output_dir)
    results_X = results.iloc[4][20]
    results_Y = results.iloc[5][20]

    forecast_dir = 'forecast_new_pos.csv'
    forecast = import_csv_to_pandas(forecast_dir)
    forecast_X = forecast.iloc[4][20]
    forecast_Y = forecast.iloc[5][20]

    plot_rose_diagram((results_X, results_Y), (forecast_X, forecast_Y))


    results_X = results.iloc[4][55]
    results_Y = results.iloc[5][55]
    forecast_X = forecast.iloc[4][55]
    forecast_Y = forecast.iloc[5][55]
    plot_rose_diagram((results_X, results_Y), (forecast_X, forecast_Y))

    results_X = results.iloc[0][55]
    results_Y = results.iloc[1][55]
    forecast_X = forecast.iloc[0][55]
    forecast_Y = forecast.iloc[1][55]
    plot_rose_diagram((results_X, results_Y), (forecast_X, forecast_Y))

    results_X = results.iloc[880][15]
    results_Y = results.iloc[881][15]
    forecast_X = forecast.iloc[880][15]
    forecast_Y = forecast.iloc[881][15]
    plot_rose_diagram((results_X, results_Y), (forecast_X, forecast_Y))

    print('check')
