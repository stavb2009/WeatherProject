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
        r: response_size - number of time steps to predict
        l: maximum length to slide - (#observation - w)
        p: final predictors - (#predictors * w)
        '''
        self.w = window_size
        self.o = 0
        self.r = 1
        self.l = 0
        self.p = 0
        self.names = []

    def re_init(self, arr):
        '''
        Helper function to initializate to 0 a vector
        '''
        arr = np.cumsum(arr)
        return arr - arr[0]

    def collect_windows(self, X, window_size=5, offset=0, previous_y=False):
        '''
        Input: X is the input matrix, each column is a variable
        Returns: diferent mappings window-output
        '''
        cols = len(list(X)) - 1
        N = len(X)

        self.o = offset
        self.w = window_size
        self.l = N - (self.w + self.r) + 1
        if not previous_y: self.p = cols * (self.w)
        if previous_y: self.p = (cols + 1) * (self.w)

        # Create the names of the variables in the window
        # Check first if we need to create that for the response itself
        if previous_y: x = cp.deepcopy(X)
        if not previous_y: x = X.drop(X.columns[-1], axis=1)

        for j, col in enumerate(list(x)):

            for i in range(self.w):
                name = col + ('(%d)' % (i + 1))
                self.names.append(name)

        # Incorporate the timestamps where we want to predict
        for k in range(self.r):
            name = '∆t' + ('(%d)' % (self.w + k + 1))
            self.names.append(name)

        self.names.append('Y')

        df = pd.DataFrame(np.zeros(shape=(self.l, (self.p + self.r + 1))),
                          columns=self.names)

        # Populate by rows in the new dataframe
        for i in range(self.l):

            slices = np.array([])

            # Flatten the lags of predictors
            for p in range(x.shape[1]):

                line = X.values[i:self.w + i, p]
                # Reinitialization at every window for ∆T
                if p == 0: line = self.re_init(line)

                # Concatenate the lines in one slice
                slices = np.concatenate((slices, line))

                # Incorporate the timestamps where we want to predict
            line = np.array([self.re_init(X.values[i:i + self.w + self.r, 0])[-1]])
            y = np.array(X.values[self.w + i + self.r - 1, -1]).reshape(1, )
            slices = np.concatenate((slices, line, y))

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


data_set = df3["2019-12-12":"2019-12-20"]
data_set.set_index('∆T', inplace=True)
w = 432
train_constructor = WindowSlider(window_size=w)
data_set = df3["2019-12-12"].copy(deep=True)
data_set.set_index('∆T', inplace=True)
train_windows = train_constructor.collect_windows(data_set.iloc[:,1:], window_size=w, offset=w, previous_y=False)

# train_windows = train_constructor.collect_windows(trainset.iloc[:, 1:],
#                                                   previous_y=False)
print("aaa")
# df3.loc['2021-08-01':'2021-08-31'] To get certain rows

#
# N = 600
#
# t = np.arange(0, N, 1).reshape(-1,1)
# t = np.array([t[i] + np.random.rand(1)/4 for i in range(len(t))])
# t = np.array([t[i] - np.random.rand(1)/7 for i in range(len(t))])
# t = np.array(np.round(t, 2))
#
# x1 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
# x2 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
# x3 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
#
# n = np.round((np.random.random(N) * 2).reshape(-1,1), 2)
#
# y = np.array([((np.log(np.abs(2 + x1[t])) - x2[t-1]**2) + 0.02*x3[t-3]*np.exp(x1[t-1])) for t in range(len(t))])
# y = np.round(y+n, 2)
#
# dataset = pd.DataFrame(np.concatenate((t, x1, x2, x3, y), axis=1),
#                        columns=['t', 'x1', 'x2', 'x3', 'y'])
#
# deltaT = np.array([(dataset.t[i + 1] - dataset.t[i]).components.minutes for i in range(len(dataset)-1)])
# deltaT = np.concatenate((np.array([0]), deltaT))
#
# dataset.insert(1, '∆t', deltaT)
# dataset.head(3)

