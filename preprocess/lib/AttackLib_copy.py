#! /bin/python3
import math
import os
import pandas as pd
import numpy as np
import json 
import random

# attack type enumeration
attack_types = {
                "benign":0,
                "random_poistion_offset":1,
                "constant_poistion_offset":2
                }


class AttackModel():
    def __init__(self,df,cfg,template):
        self.df = df.copy()
        self.cfg = cfg
        self.template = template
        
        if self.cfg['duration'] == 'random':
            # 50 length steps
            self.cfg['duration'] = np.random.randint(50) 
        
        
    def inject_attack(self):
        new_df = pd.DataFrame()
        match self.cfg['type']:
            case "random_poistion_offset":
                new_df = self.random_pos_offset(self.df)
            
            case "constant_poistion_offset":
                new_df = self.const_pos_offset(self.df)
            
            case _:
                new_df = self.df
        
        return new_df

    # case random pos offset attack
    def random_pos_offset(self,df):
        global attack_types
        tmp_df_list = []
        for _,v in df.groupby(by=['Id']):
            tmp = v.copy().reset_index(drop=True)
            # generate attacked trajectories
            if np.random.rand(1) < self.cfg['attack_rate']:
                if self.template['insert'] == 'random':
                    start = np.random.randint(int(len(tmp)-self.cfg['duration'])+1)
                else:
                    start = self.template['insert']
                    
                # position X modification
                noise_x = self.template['x_min'] + np.random.choice([-1,1],size=self.cfg['duration']
                                            )*np.random.rand(self.cfg['duration'])*np.array((self.template['x_max']-self.template['x_min']))
                
                tmp.loc[start:start+self.cfg['duration']-1,'positionX'] = tmp.loc[start:start+self.cfg['duration']-1,'positionX'] + noise_x

                # position Y modification
                noise_y = self.template['y_min'] + np.random.choice([-1,1],size=self.cfg['duration']
                                            )*np.random.rand(self.cfg['duration'])*(self.template['y_max']-self.template['y_min'])
                tmp.loc[start:start+self.cfg['duration']-1,'positionY'] = tmp.loc[start:start+self.cfg['duration']-1,'positionY'] + noise_y        


                # speed X modification
                noise_x = self.template['vx_min'] + np.random.choice([-1,1],size=self.cfg['duration']
                                            )*np.random.rand(self.cfg['duration'])*np.array((self.template['vx_max']-self.template['vx_min']))
                tmp.loc[start:start+self.cfg['duration']-1,'VelX'] = tmp.loc[start:start+self.cfg['duration']-1,'VelX'] + noise_x

                # speed Y modification
                noise_y = self.template['vy_min'] + np.random.choice([-1,1],size=self.cfg['duration']
                                            )*np.random.rand(self.cfg['duration'])*(self.template['vy_max']-self.template['vy_min'])
                tmp.loc[start:start+self.cfg['duration']-1,'VelY'] = tmp.loc[start:start+self.cfg['duration']-1,'VelY'] + noise_y


                    
                tmp['attacked'] = np.repeat(attack_types[self.cfg['type']],len(tmp))
            else:
                tmp['attacked'] = np.repeat(attack_types["benign"],len(tmp))
            
            tmp_df_list.append(tmp)
        attacked_df = pd.concat(tmp_df_list, ignore_index=True)
        return attacked_df
    
    
    
    # case constant position offset attack
    def const_pos_offset(self,df):
        global attack_types
        tmp_df_list = []  
        for _,v in df.groupby(by=['Id']):
            tmp = v.copy().reset_index(drop=True)
            # generate attacked trajectories
            if np.random.rand(1) < self.cfg['attack_rate']:
                if self.template['insert'] == 'random':
                    start = np.random.randint(int(len(tmp)-self.cfg['duration'])+1)
                else:
                    start = self.template['insert']
                
                if self.template['mode'] == 'random':
                    # position X modification
                    noise_x = np.repeat(np.random.randint(self.template['x']),self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'positionX'] = tmp.loc[start:start+self.cfg['duration']-1,'positionX'] + noise_x

                    # # position Y modification
                    noise_y = np.repeat(np.random.randint(self.template['y']),self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'positionY'] = tmp.loc[start:start+self.cfg['duration']-1,'positionY'] + noise_y        


                    # speed X modification
                    noise_x = np.repeat(np.random.randint(self.template['vx']),self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'VelX'] = tmp.loc[start:start+self.cfg['duration']-1,'VelX'] + noise_x

                    # speed Y modification
                    noise_y = np.repeat(np.random.randint(self.template['vy']),self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'VelY'] = tmp.loc[start:start+self.cfg['duration']-1,'VelY'] + noise_y      
                    
                    
                elif self.template['mode'] == 'fixed':
                    # position X modification
                    noise_x = np.repeat(self.template['x'],self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'positionX'] = tmp.loc[start:start+self.cfg['duration']-1,'positionX'] + noise_x

                    # # position Y modification
                    noise_y = np.repeat(self.template['y'],self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'positionY'] = tmp.loc[start:start+self.cfg['duration']-1,'positionY'] + noise_y        


                    # speed X modification
                    noise_x = np.repeat(self.template['vx'],self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'VelX'] = tmp.loc[start:start+self.cfg['duration']-1,'VelX'] + noise_x

                    # speed Y modification
                    noise_y = np.repeat(self.template['vy'],self.cfg['duration'])
                    tmp.loc[start:start+self.cfg['duration']-1,'VelY'] = tmp.loc[start:start+self.cfg['duration']-1,'VelY'] + noise_y


                    
                tmp['attacked'] = np.repeat(attack_types[self.cfg['type']],len(tmp))
            else:
                tmp['attacked'] = np.repeat(attack_types["benign"],len(tmp))
            
            tmp_df_list.append(tmp)
        attacked_df = pd.concat(tmp_df_list, ignore_index=True)
        return attacked_df