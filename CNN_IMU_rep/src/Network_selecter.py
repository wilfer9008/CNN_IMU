'''
Created on Mar 28, 2018

@author: fmoya
'''


import numpy as np
import argparse
import cPickle as cp
import os

from Network_user import Network_user

class Network_selecter(object):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''

        
        self.logger = None
        self.sliding_window_length = config['sliding_window_length']
        self.NB_sensor_channels = config['NB_sensor_channels']
        self.num_filters = config['num_filters']
        self.filter_size = config['filter_size']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.num_classes = config['num_classes']
        self.train_show = config['train_show']
        self.valid_show = config['valid_show']
        self.batch_size = config['batch_size']
        self.sliding_window_step = config['sliding_window_step']
        self.attrs_0 = None
        self.plotting = config['plotting']
        self.usage_modus = config['usage_modus']
        self.dataset = config['dataset']
        self.fine_tunning = config['fine_tunning']
        self.model_fine_tuning = config['model_fine_tuning']
        
        
        self.folder_exp = config['dataset'] + '/' + config['network'] + '/' + config['output'] + '/' + config['folder_exp']
        self.folder_exp_test = config['dataset'] + '/' + config['network'] + '/' + config['output'] + '/experiment'
            

        self.folder_exp_fine_tuning = config['model_fine_tuning'] + '/' + config['network'] + '/' + config['output'] + '/experiment'
            
        self.network = config['network']
        self.plotting = config['plotting']

        print("EA: used folder {}...".format(self.folder_exp))
        print "Logger will be in {}".format('../' + self.folder_exp + '/logger_exp_{}.txt')
        
        if self.fine_tunning:
            print("EA: For fine_tuning used folder {}...".format(self.folder_exp_fine_tuning))
            self.CNN = Network_user(self.folder_exp, config, folder_exp_fine_tuning = self.folder_exp_fine_tuning, folder_exp_test = self.folder_exp_test)
        else:
            self.CNN = Network_user(self.folder_exp, config, folder_exp_test = self.folder_exp_test)



    def set_logger(self, logger):
        
        self.logger = logger
        
        self.logger.info("EA: Setting logger in folder {}...".format(self.folder_exp))
        print("EA: Setting logger in folder {}...".format(self.folder_exp))
        
        self.CNN.set_logger(logger)
        
        return


    
    
    
    def test(self):
            
                
        acc_net_val, acc_f1_val, acc_dist_val, acc_dist_f1_val, percentage_test = self.CNN.evaluating_attr(ea_itera = 0,
                                                                                                           batch_size=self.batch_size,
                                                                                                           train_test_modus = 2)
        
        
        
        self.logger.info("EA: TEsting accuracy net {} with f1 {} net_dist {} f1_dist {}".format(acc_net_val, acc_f1_val,
                                                                                                 acc_dist_val, acc_dist_f1_val))
        print("EA: TEsting accuracy net {} with f1 {} net_dist {} f1_dist {}".format(acc_net_val, acc_f1_val,
                                                                                                 acc_dist_val, acc_dist_f1_val))
        
        return
    
    
    
    

    
    
    
    def net_selecter(self):
        
            
        if self.usage_modus == 'test':
            self.test()
            
        
        return
