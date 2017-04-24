# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:37:57 2017

@author: ProInspect
"""
import numpy as np  # array operations
import pandas as pd  # time series management

#rolling mean
def compute_sma(df, window):
    m = df.rolling(window=window,center=False).mean()
    return m   

def compute_sdev(df, window):
    m = df.rolling(window=window,center=False).std()
    return m   

class HoldingStock:
    No = 0
    Long = 1
    Short = 2
    
class TradingEnv:
    # Pandas data frame
    def __init__(self, df):
        # factors
        # f1: price/sma, 0 between +/- p%, 1 below 1-p, 2 above 1+p
        # f2: bollimger,  0 between  +/- 2s, 1 below, 2 above
        # f3: holding stock, 0 neutral, 1 short, 2 long
        # f4: returns since entry; 0 between +/- p%, 1 below 1-p, 2 above 1+p
        # states: n=3 obejects (0,1, 2) taken k=4 times (factors) = n^k states = 3^4 = 81 states
        # an integer of 4  with f1f2f3f4 combination values
        self.observation_space = 81
        # nothing buy sell, 0, 1, 2
        self.action_space = 3 
        # data
        self.holding_stock =  HoldingStock.No
        self.returns_since_entry = 0
        self.returns_since_entry_target = 0.1
        self.step_pos = 0
        self.df = df
        self.states = np.zeros([self.observation_space], dtype=np.int)
        self.window=50
        self.dframe = 0
        self.f1_limits = {'low':0.99, 'high':1.01}
        self.f2_limits = {'low':-0.9, 'high':0.9}
        self.f4_limits = {'low':-0.01, 'high':0.01}
        self.state_keys = {}
        self.create_table()

    def init(self, df):
        df0 = df.to_frame()
        # sma
        state_price_sma = self.price_sma(df, self.window)       
        #print(type(state_price_sma))
        #print(state_price_sma.columns.values[0])
        state_price_sma=state_price_sma.to_frame()
        state_price_sma=state_price_sma.rename(columns={state_price_sma.columns.values[0]: 'psma'})
        df0=df0.join(state_price_sma)
        # bollinger
        state_bollinger = self.bollinger(df, 20)
        state_bollinger=state_bollinger.to_frame()
        state_bollinger=state_bollinger.rename(columns={state_bollinger.columns.values[0]: 'bol'})
        df0=df0.join(state_bollinger)
        # holding
        df0['hold'] = pd.Series(0, index=df0.index)
        # returns since entry
        df0['ret_entry'] = pd.Series(0, index=df0.index)
        # returns
        ret = df/df.shift(1) - 1
        ret[0:1]=0
        ret=ret.to_frame()
        ret=ret.rename(columns={ret.columns.values[0]: 'ret'})
        df0=df0.join(ret)
        self.dframe = df0.dropna()
        return
        
    def create_table(self):
        v0 = 0
        v1 = 3
        index = 0
        for f1 in range(v0,v1):
            for f2 in range(v0,v1):
                for f3 in range(v0,v1):
                    for f4 in range(v0,v1):
                        state = f1*1000+f2*10+f3*10+f4
                        #print(state)
                        self.state_keys[state] = index
                        self.states[index] = state
                        index += 1
        
    def reset(self):
        self.holding_stock = 0
        self.returns_since_entry = 0
        self.step_pos = 0
        observation = self.state(0)
        return observation
      
    def price_sma_state(self, value):
        state = 0
        if value < self.f1_limits['low']:
            state = 1
        elif value > self.f1_limits['high']:
            state = 2
        return state

    def bollinger_state(self, value):
        state = 0
        if value < self.f2_limits['low']:
            state = 1
        elif  value > self.f2_limits['high']:
            state = 2
        return state

    def holding_state(self, value):
        return self.holding_stock

    def return_since_entry_state(self, value):
        state = 0
        if  value < self.f4_limits['low']:
            state = 1
        elif  value > self.f4_limits['high']:
            state = 2
        return state

    def get_reward(self, action, row_i):
        reward = 0
        hold = self.holding_stock
        if hold == HoldingStock.No:
            self.returns_since_entry = 0
        else:
            self.returns_since_entry +=  row_i['ret']
            #print('ret', row_i['ret'])
            reward =self.returns_since_entry
            if hold == HoldingStock.Short:
                reward =-self.returns_since_entry

        # open position
        if action == HoldingStock.Long and self.holding_stock == HoldingStock.No:
            self.holding_stock = HoldingStock.Long
            self.returns_since_entry = 0
        elif action == HoldingStock.Short and self.holding_stock == HoldingStock.No:
            self.holding_stock = HoldingStock.Short
            self.returns_since_entry = 0           
        # close position
        elif action == HoldingStock.Short and self.holding_stock == HoldingStock.Long:
            self.holding_stock = HoldingStock.No
            self.returns_since_entry = 0
        elif action == HoldingStock.Long and self.holding_stock == HoldingStock.Short:
            self.holding_stock = HoldingStock.No
            self.returns_since_entry = 0
        #print('reward', reward)
        return reward
    
    def state(self, step_pos):       
        row_i=self.dframe.iloc[step_pos,:]
        f1v = row_i['psma']
        f2v = row_i['bol']
        f3v = row_i['hold']
        f4v = row_i['ret_entry']
        s1 =self. price_sma_state(f1v)
        s2 =self. bollinger_state(f2v)
        s3 =self. holding_state(f3v)
        s4 =self. return_since_entry_state(f4v)
        observation = s1*1000+s2*10+s3*10+s4
        s = self.state_keys[observation]
        return s, observation

    def stateCurrent(self):       
        return self.state(self.step_pos)
    
    def step(self, action): 
        if  self.step_pos < len(self.dframe):
            self.step_pos += 1
        # get the row at date step_pos
        row_i=self.dframe.iloc[self.step_pos,:]
        s, observation = self.state(self.step_pos)
        reward = self.get_reward(action, row_i)
        # 3 is hold
        self.dframe.iloc[self.step_pos,3] = self.holding_stock
        self.dframe.iloc[self.step_pos,4] = self.returns_since_entry
        f1v = row_i['psma']
        f2v = row_i['bol']
        f3v = row_i['hold']
        f4v = row_i['ret_entry']
        #done = f4v > self.returns_since_entry_target
        info = {'s': observation, 'f1': f1v, 'f2': f2v,'f3': f3v,'f4': f4v}
        done = self.step_pos+1 > len(self.dframe)
        return s, reward, done, info
        
    def price_sma(self, df, window):
        sma=compute_sma(df, window=window)
        psma=df.div(sma, axis=0)
        return psma

    def bollinger(self, df, window):
        sma=compute_sma(df, window=window)
        sdev=compute_sdev(df, window=window)
        d=df-sma
        b=d.div(2*sdev, axis=0)
        return b        