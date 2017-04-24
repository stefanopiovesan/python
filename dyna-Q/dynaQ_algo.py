# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:31:49 2017

@author: ProInspect
"""

import numpy as np  # array operations


class DynaQAlgo:

    def __init__(self, env):
        self.env = env
        #Initialize table with all zeros
        self.Q = np.zeros([env.observation_space,env.action_space])
        # Set learning parameters
        self.lr = .85
        self.y = .99
        self.num_episodes = 1000
        #create lists to contain total rewards and steps per episode
        #jList = []
        self.rList = []
        

    def run(self):
        for i in range(self.num_episodes):
            #Reset environment and get first new observation
            s, o = self.env.reset()
            rAll = 0
            d = False
            j = 0
            i = 0
            #The Q-Table learning algorithm
            while j < 99:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(self.Q[s,:] + np.random.randn(1,self.env.action_space)*(1./(i+1)))
                #Get new state and reward from environment
                s1,r,d,_ = self.env.step(a)
                #print(s1,r,d)
                #Update Q-Table with new knowledge
                self.Q[s,a] = self.Q[s,a] + self.lr*(r + self.y*np.max(self.Q[s1,:]) - self.Q[s,a])
                #print(s, a, Q[s,a])
                rAll += r
                s = s1
                if d == True:
                    break
            #jList.append(j)
            self.rList.append(rAll)
        return self.Q
            