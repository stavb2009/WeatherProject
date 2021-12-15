# Import the required packages
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from math import pi
from windrose import WindroseAxes

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    df = pd.read_csv(r'C:\Users\stav2\PycharmProjects\pythonProject3\Og_10Mi.csv', engine='python', skipfooter=3)
    df['wsp_num']=pd.to_numeric(df['Wsp_WS4_Avg'],errors='coerce')
    df['wdr_num'] = pd.to_numeric(df['Wdr_WS4_Avg'], errors='coerce')
    I_list = [i for i, v in enumerate(df['wsp_num']) if v > 30]
    df['wsp_num'][I_list] = 0
    df['WDR_x'] = df['wsp_num'] * np.sin(df['wdr_num'] * pi / 180.0)
    df['WDR_y'] = df['wsp_num'] * np.cos(df['wdr_num'] * pi / 180.0)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect('equal')
    df.plot(kind='scatter', x='WDR_x', y='WDR_y', alpha=0.35, ax=ax)
    # < AxesSubplot: xlabel = 'WDR_x', ylabel = 'WDR_y' >
    round(df['wsp_num']).hist(figsize=(20, 9))
    ax = WindroseAxes.from_ax()
    ax.bar(df.wdr_num, df.wsp_num, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    print("aa")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


