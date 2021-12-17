import numpy as np
import pandas as pd
from datetime import datetime
import re
import time

"""
I took all of that from this site:
https://towardsdatascience.com/ml-approaches-for-time-series-4d44722e48fe
"""


class WindowSlider(object):

    def __init__(self, window_size=5):
        '''
        Window Slider object
        ====================
        w: window_size - number of time steps to look back
        o: offset between last reading and temperature
        s: response_size - number of time steps to predict
        l: length to slide - ((#observation - w)/s)+1
        '''
        self.w = window_size
        self.s = 1
        self.l = 0
        self.names = []

    def re_init(self, arr):
        '''
        Helper function to initializate to 0 a vector
        '''
        arr = np.cumsum(arr)
        return arr - arr[0]

    def collect_windows(self, X, window_size=5, step=1):
        '''
        Input: X is the input matrix, each column is a variable
        Returns: diferent mappings window-output
        '''
        cols = len(list(X))
        N = len(X)

        self.w = window_size
        self.s = step
        self.l = int(np.floor((N -self.w)/self.s) + 1)

        if self.l<0:
            print("Class: WindowSlider, def: collect_windows, error: Length od input data is too short for the window")
            return -1

        # Create the names of the variables in the window
        # Check first if we need to create that for the response itself

        self.names.append(X.index.name)

        for j, col in enumerate(list(X)):

            for i in range(self.w):
                name = col + ('(%d)' % (i + 1))
                self.names.append(name)
        #
        # # Incorporate the timestamps where we want to predict
        # for k in range(self.s):
        #     name = '∆t' + ('(%d)' % (self.w + k + 1))
        #     self.names.append(name)


        df = pd.DataFrame(np.zeros(shape=(self.l,len(self.names))),
                          columns=self.names)

        # Populate by rows in the new dataframe
        for i in range(self.l):

            slices = np.array([X.index[i*self.w]])

            # Flatten the lags of predictors
            for p in range(X.shape[1]):

                line = X.values[i*self.s:self.w + i*self.s, p]



                # Concatenate the lines in one slice
                slices = np.concatenate((slices, line))

                # Incorporate the timestamps where we want to predict

            # Incorporate the slice to the cake (df)
            df.iloc[i, :] = slices

        return df


def dataset_prep(dataset_location=None, name_of_dates_column=None, i=None):

    if not dataset_location:
        print("No DataSet location have inserted")
        return -1

    df = pd.read_csv(dataset_location)

    dates = df[name_of_dates_column]
    list_times = [dates[i].split(' ') for i in range(2, len(dates))]
    days, hour_of_day = zip(*list_times)
    #

    hour_of_day_format = ['0'+sub if re.search("^.:", sub) else sub for sub in hour_of_day]
    minute_of_day = [(int(hour_of_day[i].split(':')[0]))*60 + (int(hour_of_day[i].split(':')[1])) for i in range(len(hour_of_day))]
    days2 = [days[i].split('/') for i in range(len(days))]
    month, day, year = zip(*days2)
    month_format = ['0'+sub if len(sub) < 2 else sub for sub in month]
    time_format = [datetime.fromisoformat(year[i] + '-' + month_format[i] + '-' + day[i] + ' ' + hour_of_day_format[i]) for i in range(len(days))]
    day = [int(day[i]) for i in range(len(day))]
    month = [int(month[i]) for i in range(len(month))]
    year = [int(year[i]) for i in range(len(year))]

    df = df[2::].copy(deep=True)
    df["minute_of_day"] = minute_of_day
    df["day"] = day
    df["month"] = month
    df["year"] = year
    df["Date_format"] = time_format
    df.set_index('Date_format', inplace=True)
    return df

dataset_location = 'data\\dead_see_weather_dates_edited.csv'
df = pd.read_csv(dataset_location)

df3 = dataset_prep(dataset_location, name_of_dates_column='TIMESTAMP')

deltaT = np.array([((df3.index[i + 1] - df3.index[i]).value)/1e10 for i in range(len(df3)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))

df3.insert(2, '∆T', deltaT)


data_set = df3["2018-12-12":"2018-12-20"]
data_set.drop(columns=['TIMESTAMP', 'minute_of_day', 'day', 'month', 'year'], axis=1, inplace=True)

w = 5
train_constructor = WindowSlider(window_size=w)


train_windows = train_constructor.collect_windows(data_set, window_size=w, step=w)

print("aaa")
# df3.loc['2021-08-01':'2021-08-31'] To get certain rows


