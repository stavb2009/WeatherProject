#@title Default title text
import sys
import pandas as pd
import numpy as np
import os
import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
import tensorflow as tf
import math

def add_forcast(data, forcast, data_len=24, jump=3, total_len=48, resolution = 10):
    """
    :param data: single data set of observation
    :param forcast: single data set of model prediction
    :param dim: default 'col', get 'col' or 'row'. dim of time progression
    :param jump: diff between raw forcast data in minutes. default 3 hours
    :param resolution: data resolution (time diff between each unit) in minutes. default 10 min
    :param total_len: total length (in minutes) of desired output spaced forecast. default 12 hours
    :return:
    """
    data_len=data_len*60//resolution
    jump=jump*60//resolution
    total_len=total_len*60//resolution
    for i in range(data_len, total_len, jump):  # for example : fill every 3h, according
        # to jump
      #print(i)
      try:
          
          data.iloc[:,i] = forcast.iloc[:,i-data_len] 

      except:
          pass
    return data

# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model): # in our case, position = n tokens, d_model=1
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
    
    def __init__(self, data, forcast,threshold=20,time_col='',wind_col='',
                 data_col='',jumps=48,resolution=10,test=0,total_len='',pipeline=1,drop =1):  #done
        """
        deletes unwanted rows and resets rows indices afterward
        for now, data and forecast are panda table. can be changed
        :param jumps:time jumps between block. default 48 hours
        :param resolution:reading resolution. default 10 min
        :param total_len: len (in hours) of the data we want to insert the machine
        """
        self.data = data
        self.df = data
        self.data_shape = self.df.shape
        self.forcast = forcast
        self.forcast_shape = self.forcast.shape
        self.val = pd.DataFrame()
        self.max = 0
        self.test=test
        cols = data.columns 
        if not wind_col: self.wind_col=['Wsp_WS4_Avg','Wdr_WS4_Avg']
        else:self.wind_col=wind_col
        if not time_col: self.time_col='TIMESTAMP'
        else:self.time_col=time_col
        if not data_col: self.data_col=['T_DL_Avg']
        else:self.data_col=data_col
        if not total_len: self.total_len = 48
        else: self.total_len = total_len
        if drop:
          for i in range(0,self.data_shape[0]):
            try:
                x = float(self.data.iloc[i][1])
            except:
                try:
                  if math.isnan(float(self.data.iloc[i][1])): 
                    self.data.drop(range(0,i),inplace=True)
                except: pass
            else: 
                self.data.drop(range(0,i),inplace=True)
                break
          self.data.reset_index(drop=True,inplace=True)
        self.max=[]
        self.jumps=jumps
        self.resolution = resolution
        self.threshold=threshold
        self.d_model= 4  # default, only timestamp,wy,wx,Temperature
        if pipeline: self.pipeline()

    def point2xy(self, ws_name, wd_name, dim='col'): #done
        """
        convert ws and wd to xy vector
        :param ws_name: ws column name
        :param wd_name: wd column name
        :param dim: 'col' to change columns, 'row' to change rows. default 'col'
        :return: same dataset, but with 2 new columns/rows (x,y)  
        """
        ## to do: check if it's the same for model data
        if dim == 'col':
            print(ws_name,wd_name)
            ws=pd.to_numeric(self.data[ws_name], errors='ignore')
            wd=pd.to_numeric(self.data[wd_name], errors='ignore')
            wx = ws * np.sin(wd)
            wy = ws * np.cos(wd)
            self.data[ws_name] = wx
            self.data[wd_name] = wy
            self.data.rename(columns={ws_name: 'wx', wd_name: 'wy'}, inplace=True)
            self.wind_col=['wx','wy']
        elif dim == 'row':
            ws=pd.to_numeric(self.data.iloc[ws_name], errors='ignore')
            wd=pd.to_numeric(self.data.iloc[wd_name], errors='ignore')
            wx = ws * np.sin(wd)
            wy = ws * np.cos(wd)
            self.data.iloc[ws_name] = wx
            self.data.iloc[wd_name] = wy
            self.data.rename(index={ws_name: 'wx', wd_name: 'wy'}, inplace=True)
        if(self.test) : print("point2xy")
            
    
    def pipeline(self):
      self.point2xy(self.wind_col[0],self.wind_col[1])
      self.normlize(self.wind_col[0],tanh=1)
      self.normlize(self.wind_col[1],tanh=1)
      if self.data_col:
        for col in self.data_col:
          self.normlize(col,oulier=0)
      self.delete_junk()
      self.organize_data()
      if(self.test) : print("pipeline")


    def xy2point(self, xname, yname, dim='col'):  #done
        """
        convert ws and wd to xy vector
        :param xname: x component column name
        :param yname: y component column name
        :param dim: 'col' to change columns, 'row' to change rows. default 'col'
        :return: same dataset, but with 2 new columns/rows (ws,wd)  
        """
        ## to do: check if it's the same for model data
        if dim == 'col':
            x=pd.to_numeric(self.data[xname], errors='ignore')
            y=pd.to_numeric(self.data[yname], errors='ignore')
            ws = np.sqrt(x**2 + y**2)**0.5
            wd = np.arctan(y/x)
            self.data[xname] = ws
            self.data[yname] = wd
            self.data.rename(columns={xname: 'ws', yname: 'dy'}, inplace=True)
        elif dim == 'row':
            x=pd.to_numeric(self.data.iloc[xname], errors='ignore')
            y=pd.to_numeric(self.data.iloc[yname], errors='ignore')
            ws = np.sqrt(x**2 + y**2)**0.5
            wd = np.arctan(y/x)
            self.data.iloc[xname] = ws
            self.data.iloc[yname] = wd
        if(self.test) : print("xy2point")

    def normlize(self, colname, oulier=1,tanh=0):  #done
        """
        normlize columns to [-1,1]
        :param xname: x component column name
        :param yname: y component column name
        :param dim: 'col' to change columns, 'row' to change rows. default 'col'
        :return: same dataset, but with 2 new columns/rows (ws,wd)  
        """
        ## to do: check if it's the same for model data
        self.data[colname]=pd.to_numeric(self.data[colname], errors='ignore')
        if oulier: 
          cond=self.data.index[(abs(self.data[colname]))>self.threshold].tolist()
          self.data.drop(cond,inplace=True)
        
        df=self.data[colname]
        self.max.append(abs(df).max())
        self.normalized_df=(df/abs(df).max())
        if tanh : self.tanh (3)
        self.data[colname] = self.normalized_df 
        if(self.test) : print("normlize")
        
    def tanh(self,factor):
        for i in range(len(self.normalized_df)): 
          try:
            self.normalized_df[i] = math.tanh(factor*self.normalized_df[i])
          except:
            pass
      


    def fix_dates(self): #done
        """
        :param date_col: date's column name
        :return: change the dates to datatime format
        """
        for i in range(0,len(self.data)):
          t = (self.data[self.time_col]).iloc[i]
          if (type(t) is not datetime.datetime):
            t = datetime.datetime.strptime(t, '%m/%d/%Y %H:%M')
            (self.data[self.time_col]).iloc[i]=t
        if(self.test) : print("fix_dates")
        

    def delete_junk(self, starting_time='first val'):
        """
        :param starting_time: the starting time of each block, 
        in int(hh,mm) format. default the first hour in the table
        
        :return: data table without ****
        """
        
        self.fix_dates()  #change dates to datatime format
        if starting_time == 'first val':
          starting_date=self.data.iloc[0,0]
          self.starting_time=(starting_date.hour,starting_date.minute)
        else:
          self.starting_time=starting_time  
        res_delta = timedelta(minutes=self.resolution)
        jump_delta = timedelta(hours=48)
        i=0
        jumps_index = int(self.jumps*60/self.resolution)
        if(self.test) : print("delete_junk start")
        while (i < len(self.data)-1):
          try:
            t=self.data.iloc[i+jumps_index,0]
          except:
            break
          else:
            if (t.hour,t.minute) != self.starting_time:
              self.data.drop(self.data.index[i],inplace=True)
              while (self.data.iloc[i,0].hour,self.data.iloc[i,0].minute) != self.starting_time:
                self.data.drop(self.data.index[i],inplace=True)
              #print("drop", i)
            else:
              i=i+jumps_index
          self.data.reset_index(drop=True,inplace=True)
        if(self.test) : print("delete_junk end")
            
    def organize_data(self):
      """
        :param jumps:time jumps between block. default 48 hours
        :param resolution:reading resolution. default 10 min
        :param cols: column names to include new table.
        :return: data table stacked, ready for machine
      """
      
      cols=[self.time_col]+self.wind_col+self.data_col
      self.d_model  = len(cols)
      index_jump=int(self.jumps*60/self.resolution)
      a = range(0,len(self.data),index_jump)
      self.stacked_data=pd.DataFrame()
      reduced_data=self.data[cols]
      s0=reduced_data.iloc[a[0]:a[0+1]]
      indices=s0.index.tolist()
      for i in range(len(a)):
        try:
          s=reduced_data.iloc[a[i]:a[i+1]]
          s=s.transpose()
          s.set_axis(indices, axis=1,inplace=True)
          frames = [self.stacked_data, s]
          self.stacked_data = pd.concat(frames)
        except:
          pass
      if(self.test) : print("organize_data end")

    def math2met(self, colname, dim='col'):
        """
        :param dim:
        :param dirname: string of U component column's name in dt
        :return: change the wd from math wind to meteorological wind, or backward
        """
        if dim == 'col':
            self.data[colname] = self.df[colname] - 270
        elif dim == 'row':
            self.data.loc[colname] = self.df.loc[colname] - 270

    def find_closest(self,forcast_times,time):
            """
            :param time:
            :param forcast_times: pandas list of model starting times
            :return: index and value of the closest model run before this time
            """
            time_list = forcast_times.tolist()
            
            time_list_reduced = forcast_times[forcast_times>time].tolist()
            
            min_val = min(time_list_reduced, key=lambda x: abs(x - time) )
            min_index = time_list.index(min_val)
            return min_val, min_index

    def spacing(self,forecast_jumps=3):
        """
        :param forcast_df:
        :param len:
        :param jump:
        :param resolution:
        :return:
        """
        self.data.dropna(axis=1, how='any', inplace=True)
        data_len=self.data.shape[1]  ## before change
       
        ## padding with nans
        data_shape = self.data.shape
        forcast_shape = self.forcast.shape
        data_extended_range = np.arange(self.total_len*60 // self.resolution)  # num of data units in desired output
        self.data = self.data.reindex(columns=data_extended_range)

        time_indices = np.arange(0, forcast_shape[0], forecast_jumps)
        forcast_time_array = self.forcast.iloc[np.arange(0, forcast_shape[0], self.d_model), 0]
        #print(forcast_time_array)
        # data_time_array = data.iloc[np.arange(0, data[0], 3), 0]
        tokens = self.total_len*60 // self.resolution
        dimensions = 1
        pos_encoding = positional_encoding(tokens, dimensions)
        for i in range(0, self.data.shape[0],self.d_model):
        #for i in range(0, 1): ## test

            #print("spacing i=",i)
            val, closest_index = self.find_closest(forcast_time_array,self.data[0][i])
            #print("val, closest_index =" ,val, closest_index )
            spaced_data = add_forcast(self.data.iloc[i:i + self.d_model],
                          self.forcast.iloc[self.d_model*closest_index:self.d_model*(closest_index+1)],
                          data_len=24, jump=3)
            #print(spaced_data.T)
            for d in range(1,self.d_model):
                spaced_data.iloc[d] = spaced_data.iloc[d] + pos_encoding[0, :, 0]
            self.data.iloc[i:i + self.d_model] = spaced_data
        self.data.dropna(axis=1, how='any', inplace=True)


    def denormlize(self,colname,max):
        """
        retur columns to previous scale
        """
        ## to do: check if it's the same for model data
        cond=self.data.index[(abs(self.data[colname]))>self.threshold].tolist()
        self.data.drop(cond,inplace=True)
        df=self.data[colname]
        denormalized_df=(df*max)
        self.data[colname] = denormalized_df 
        if (self.test) :print("denormlize")





