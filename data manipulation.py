# import sys
import pandas as pd
import numpy as np
import os
import datetime
from datetime import datetime


def add_forcast(data, forcast, data_len, dim='row', jump=3, total_len=70):
    """
    :param data: single data set of observation
    :param forcast: single data set of model prediction
    :param dim: default 'col', get 'col' or 'row'. dim of time progression
    :param jump: diff between raw forcast data in minutes. default 3 hours
    :param resolution: data resolution (time diff between each unit) in minutes. default 10 min
    :param total_len: total length (in minutes) of desired output spaced forecast. default 12 hours
    :return:
    """
    data_shape = data.shape
    forcast_shape = forcast.shape
    # df_range = np.arange(total_len // resolution)  # num of data units in desired output
    if dim == 'col':
        # data = data.reindex(df_range)
        j = 0
        for i in range(data_shape[0], total_len, jump):  # for example : fill every 3h, according
            # to jump
            try:
                data.iloc[i] = forcast.iloc[j]  # 0 is a dummy value for now
                j = j + 1
            except:
                pass

    elif dim == 'row':
        # print("data now ",data,"\n")
        # data = data.reindex(columns=df_range)
        # print("new data: ",data)
        j = 0
        for i in range(data_len, total_len, jump):  # for example : fill every 3h, according
            # to jump
            try:
                data[i] = forcast[j]
                j = j + 1
            except:
                pass
    #    print("\n data from single : \n",data)
    return data


# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding.transpose()


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


class Modifier:
    def __init__(self, df, forcast, total_len):
        self.df = df
        self.shape = self.df.shape
        self.forcast = forcast
        self.total_len = total_len
        self.val = pd.DataFrame()
        self.max = 0

    def point2xy(self, ws, wd, dim='col'):
        """
        :param wd:
        :param ws:
        :param dim: 'col' to change columns, 'row' to change rows
        :param xname: string of X coordinate column's/row's name in self.df
        :param yname: string of y coordinate column's/row's name in self.df
        :return: same dataset, but with 2 new columns/rows (ws,wd) instead of xname yname
        """
        if dim == 'col':
            wx = self.df[ws] * np.sin(self.df[wd])
            wy = self.df[ws] * np.cos(self.df[wd])
            self.df[ws] = wx
            self.df[wd] = wy
            self.df.rename(columns={ws: 'wx', wd: 'wy'}, inplace=True)
        elif dim == 'row':
            ws = (self.df.loc[ws] ** 2 + self.df.loc[wd] ** 2) ** 0.5
            wd = np.degrees(np.arctan2(self.df.loc[wd], self.df.loc[ws]))
            self.df.loc[ws] = ws
            self.df.loc[wd] = wd
            self.df.rename(index={ws: 'ws', wd: 'wd'}, inplace=True)

    def xy2point(self, xname, yname, dim='col'):
        """
        :param self:
        :param dim: 'col' to change columns, 'row' to change rows
        :param xname: string of X coordinate column's/row's name in df
        :param yname: string of y coordinate column's/row's name in df
        :return: same dataset, but with 2 new columns/rows (ws,wd) instead of xname yname
        """
        if dim == 'col':
            self.df[xname] = pd.to_numeric(self.df[xname], errors='coerce')
            self.df[yname] = pd.to_numeric(self.df[yname], errors='coerce')
            ws = (self.df[xname] ** 2 + self.df[yname] ** 2) ** 0.5
            wd = np.degrees(np.arctan2(self.df[yname], self.df[xname]))
            self.df[xname] = ws
            self.df[yname] = wd
            self.df.rename(columns={xname: 'ws', yname: 'wd'}, inplace=True)
        elif dim == 'row':
            ws = (self.df.loc[xname] ** 2 + self.df.loc[yname] ** 2) ** 0.5
            wd = np.degrees(np.arctan2(self.df.loc[yname], self.df.loc[xname]))
            self.df.loc[xname] = ws
            self.df.loc[yname] = wd
            self.df.rename(index={xname: 'ws', yname: 'wd'}, inplace=True)

    def math2met(self, dirname, dim='col'):
        """
        :param dim:
        :param dirname: string of U component column's name in dt
        :return: change the wd from math wind to meteorological wind, or backward
        """
        if dim == 'col':
            self.df[dirname] = self.df[dirname] - 270
        elif dim == 'row':
            self.df.loc[dirname] = self.df.loc[dirname] - 270

    def find_closest(self, forcast_times, time):
        """
        :param time:
        :param forcast_times: pandas list of model starting times
        :return: index and value of the closest model run before this time
        """
        time_list = forcast_times.values.tolist()
        min_val = min(time_list, key=lambda x: abs(x - time))
        min_index = time_list.index(min_val)
        return min_val, min_index

    def df_spacing(self, jump=3 * 60, resolution=10):
        """

        :param forcast_df:
        :param len:
        :param jump:
        :param resolution:
        :return:
        """
        data_shape = self.df.shape
        forcast_shape = self.forcast.shape
        df_range = np.arange(self.total_len // resolution)  # num of data units in desired output
        data = self.df.reindex(columns=df_range, copy=False)

        time_indices = np.arange(0, forcast_shape[0], 3)
        forcast_time_array = self.forcast.iloc[np.arange(0, forcast_shape[0], 3), 0]
        # data_time_array = data.iloc[np.arange(0, data[0], 3), 0]
        tokens = self.total_len // resolution
        dimensions = 2
        pos_encoding = positional_encoding(tokens, dimensions)
        for i in range(0, data.shape[0], 3):
            # for i in range(0,3,3):
            val, closest_index = self.find_closest(data[0][6], forcast_time_array)

            spaced_data = add_forcast(data.iloc[i:i + 3],
                                      self.forcast.iloc[3 * closest_index:3 * closest_index + 3],
                                      data_len=data_shape[1], jump=jump // resolution)
            spaced_data.iloc[1:3] = spaced_data.iloc[1:3] + pos_encoding[:, :, 0]
            data.iloc[i:i + 3] = spaced_data

            #### not general, need to be fixed
            x = (data.iloc[1]).to_numpy()
            result = np.argwhere(x == x)
            self.df = data.reindex(columns=result, copy=False)

    def normlize(self, data, val):
        indices_temp = np.arange(0, data.shape[0])
        scale_fun = lambda x: x % 3 != 0
        indices = indices_temp[scale_fun(indices_temp)]
        data0 = data.pop(0)
        val0 = val.pop(0)
        A = (data.iloc[indices]).max()
        B = (val.iloc[indices]).max()
        max_val = np.maximum(A.max(), B.max())
        for index in indices:
            data.iloc[index] = data.iloc[index] / max_val
            val.iloc[index] = val.iloc[index] / max_val
        val.insert(loc=0, column='0', value=val0)
        data.insert(loc=0, column='0', value=data0)
        self.max = max

    def denormlize(self, data, val):
        indices_temp = np.arange(0, data.shape[0])
        scale_fun = lambda x: x % 3 != 0
        indices = indices_temp[scale_fun(indices_temp)]
        data0 = data.pop(0)
        val0 = val.pop(0)
        for index in indices:
            data.iloc[index] = data.iloc[index] * self.max
            val.iloc[index] = val.iloc[index] * self.max
        val.insert(loc=0, column='0', value=val0)
        data.insert(loc=0, column='0', value=data0)


def main():
    df = pd.read_excel('data.xlsx')
    print("")
    print("")


if __name__ == "__main__":
    main()
