'''
Created on Feb 28, 2017

@author: fmoya
'''


import caffe
from caffe import layers as L
from caffe import params as P

import io

class CNN(object):
    '''
    classdocs
    '''


    def __init__(self, num_classes, num_filters, output, network_type, 
                 dataset, filter_size, folder_exp, lr, fine_tunning, sliding_window_length,
                 NB_sensor_channels):
        '''
        Constructor
        '''
        
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.output = output
        self.network_type = network_type
        self.dataset = dataset
        self.filter_size = filter_size
        self.folder_exp = folder_exp
        self.lr = lr
        self.fine_tunning = fine_tunning
        self.sliding_window_length = sliding_window_length
        self.NB_sensor_channels = NB_sensor_channels

        self.weight_param = dict(lr_mult=1, decay_mult=1)
        self.bias_param   = dict(lr_mult=1, decay_mult=0)
        self.learned_param = [self.weight_param, self.bias_param]
        #self.frozen_param = [dict(lr_mult=0)] * 2
        self.frozen_param = [self.weight_param, self.bias_param]
        
    def network_cnn(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        n = caffe.NetSpec()
        
        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.data, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,self.NB_sensor_channels]),
                                               dict(dim=[batch_size, 1])], ntop = 2)

        n.conv1 = L.Convolution(n.data, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1 = L.ReLU(n.conv1, in_place=True)
        n.norm1 = L.LRN(n.relu1, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2 = L.Convolution(n.norm1, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2 = L.ReLU(n.conv2, in_place=True)
        
        n.pool2 = L.Pooling(n.relu2, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)
        
        
        n.conv3 = L.Convolution(n.pool2, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3 = L.ReLU(n.conv3, in_place=True)
        n.conv4 = L.Convolution(n.relu3, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4 = L.ReLU(n.conv4, in_place=True)
        
        n.pool4 = L.Pooling(n.relu4, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)
        
        if if_Train:
            n.drop4 = fc5input  = L.Dropout(n.pool4, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input = n.pool4
        
        #n.drop4  = L.Dropout(n.relu4, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN)) 

        n.fc5 = L.InnerProduct(fc5input, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_relu = L.ReLU(n.fc5, in_place=True)
        
        


        if use_maxout:
            if if_Train:
                n.droppremax = L.Dropout(n.fc5_relu, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size, int(512/2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.fc5_relu, shape=dict(dim=[batch_size,int(512/2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        else:
            
            if if_Train:
                n.drop5 = fc6input  = L.Dropout(n.fc5_relu, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
            else:
                fc6input = n.fc5_relu
        
            n.fc6 = L.InnerProduct(fc6input, num_output=256, param = self.learned_param, 
                                  weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        if self.output == 'softmax':
        
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
                    
            
        
        return n.to_proto()
    




    def network_cnn_pamap2(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        n = caffe.NetSpec()
        
        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.data, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,self.NB_sensor_channels]),
                                               dict(dim=[batch_size, 1])], ntop = 2)

        n.conv1 = L.Convolution(n.data, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1 = L.ReLU(n.conv1, in_place=True)
        n.norm1 = L.LRN(n.relu1, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2 = L.Convolution(n.norm1, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2 = L.ReLU(n.conv2, in_place=True)
        n.pool2 = L.Pooling(n.relu2, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3 = L.Convolution(n.pool2, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3 = L.ReLU(n.conv3, in_place=True)
        
        
        n.conv4 = L.Convolution(n.relu3, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4 = L.ReLU(n.conv4, in_place=True)
        
        
        n.pool4 = L.Pooling(n.relu4, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        #One ends up with 744 window size
        

        '''
        n.conv5 = L.Convolution(n.pool4, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu_conv5 = L.ReLU(n.conv5, in_place=True)
        
        
        n.conv6 = L.Convolution(n.relu_conv5, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu_conv6 = L.ReLU(n.conv6, in_place=True)

        n.pool6 = L.Pooling(n.relu_conv6, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        


        '''
        

        
        if if_Train:
            n.drop4 = fc5input  = L.Dropout(n.pool4, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input = n.pool4
        
        
        #n.drop4  = L.Dropout(n.relu4, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN)) 

        n.fc5 = L.InnerProduct(fc5input, num_output=128, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_relu = L.ReLU(n.fc5, in_place=True)
        
        


        if use_maxout:
            if if_Train:
                n.droppremax = L.Dropout(n.fc5_relu, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size, int(128/2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=64, param = self.learned_param, 
                                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.fc5_relu, shape=dict(dim=[batch_size,int(128/2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=64, param = self.learned_param, 
                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        else:
            
            if if_Train:
                n.drop5 = fc6input  = L.Dropout(n.fc5_relu, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
            else:
                fc6input = n.fc5_relu
        
            n.fc6 = L.InnerProduct(fc6input, num_output=128, param = self.learned_param, 
                                  weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        
        if self.output == 'softmax':
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
                    
            
        
        return n.to_proto()
        

    def network_cc_imu_opportunity(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        
        n = caffe.NetSpec()
        

        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.acce, n.back, n.rua, n.rla, n.lua, n.lla, n.lshoe, n.rshoe, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,36]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,9]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,9]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,9]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,9]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,9]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,16]),
                                                                                                     dict(dim=[batch_size,1,self.sliding_window_length,16]),
                                                                                                     dict(dim=[batch_size, 1])], ntop = 9)
        
        # For acce

        n.conv1_acce = L.Convolution(n.acce, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_acce = L.ReLU(n.conv1_acce, in_place=True)
        n.conv2_acce = L.Convolution(n.relu1_acce, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_acce = L.ReLU(n.conv2_acce, in_place=True)

        n.norm2_acce = L.LRN(n.relu2_acce, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_acce = L.Pooling(n.norm2_acce, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_acce = L.Convolution(n.pool2_acce, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_acce = L.ReLU(n.conv3_acce, in_place=True)
        
        n.conv4_acce = L.Convolution(n.relu3_acce, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_acce = L.ReLU(n.conv4_acce, in_place=True)

                
        if if_Train:
            n.drop4_acce = fc5input_acce  = L.Dropout(n.relu4_acce, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_acce = n.relu4_acce


        n.fc5_acce = L.InnerProduct(fc5input_acce, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_acce_relu = L.ReLU(n.fc5_acce, in_place=True)
        
        
        

        # For back

        n.conv1_back = L.Convolution(n.back, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_back = L.ReLU(n.conv1_back, in_place=True)
        n.conv2_back = L.Convolution(n.relu1_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_back = L.ReLU(n.conv2_back, in_place=True)

        n.norm2_back = L.LRN(n.relu2_back, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_back = L.Pooling(n.norm2_back, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_back = L.Convolution(n.pool2_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_back = L.ReLU(n.conv3_back, in_place=True)

        n.conv4_back = L.Convolution(n.relu3_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_back = L.ReLU(n.conv4_back, in_place=True)
        
        if if_Train:
            n.drop4_back = fc5input_back  = L.Dropout(n.relu4_back, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_back = n.relu4_back

        n.fc5_back = L.InnerProduct(fc5input_back, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_back_relu = L.ReLU(n.fc5_back, in_place=True)
        
        
        
        
        
        # For rua

        n.conv1_rua = L.Convolution(n.rua, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_rua = L.ReLU(n.conv1_rua, in_place=True)
        n.conv2_rua = L.Convolution(n.relu1_rua, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_rua = L.ReLU(n.conv2_rua, in_place=True)

        n.norm2_rua = L.LRN(n.relu2_rua, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_rua = L.Pooling(n.norm2_rua, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_rua = L.Convolution(n.pool2_rua, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_rua = L.ReLU(n.conv3_rua, in_place=True)

        n.conv4_rua = L.Convolution(n.relu3_rua, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_rua = L.ReLU(n.conv4_rua, in_place=True)

        
        
        if if_Train:
            n.drop4_rua = fc5input_rua  = L.Dropout(n.relu4_rua, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_rua = n.relu3_rua
        

        n.fc5_rua = L.InnerProduct(fc5input_rua, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_rua_relu = L.ReLU(n.fc5_rua, in_place=True)
        



        # For rla

        n.conv1_rla = L.Convolution(n.rla, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_rla = L.ReLU(n.conv1_rla, in_place=True)
        n.conv2_rla = L.Convolution(n.relu1_rla, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_rla = L.ReLU(n.conv2_rla, in_place=True)

        n.norm2_rla = L.LRN(n.relu2_rla, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_rla = L.Pooling(n.norm2_rla, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_rla = L.Convolution(n.pool2_rla, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_rla = L.ReLU(n.conv3_rla, in_place=True)

        n.conv4_rla = L.Convolution(n.relu3_rla, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_rla = L.ReLU(n.conv4_rla, in_place=True)

        if if_Train:
            n.drop4_rla = fc5input_rla  = L.Dropout(n.relu4_rla, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_rla = n.relu4_rla
        
        n.fc5_rla = L.InnerProduct(fc5input_rla, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_rla_relu = L.ReLU(n.fc5_rla, in_place=True)
        
        
        
        
        # For lua

        n.conv1_lua = L.Convolution(n.lua, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_lua = L.ReLU(n.conv1_lua, in_place=True)
        n.conv2_lua = L.Convolution(n.relu1_lua, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_lua = L.ReLU(n.conv2_lua, in_place=True)

        n.norm2_lua = L.LRN(n.relu2_lua, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_lua = L.Pooling(n.norm2_lua, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_lua = L.Convolution(n.pool2_lua, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_lua = L.ReLU(n.conv3_lua, in_place=True)
        '''
        n.conv4_lua = L.Convolution(n.relu3_lua, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_lua = L.ReLU(n.conv4_lua, in_place=True)
        '''
        if if_Train:
            n.drop4_lua = fc5input_lua  = L.Dropout(n.relu3_lua, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_lua = n.relu3_lua
        
        n.fc5_lua = L.InnerProduct(fc5input_lua, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_lua_relu = L.ReLU(n.fc5_lua, in_place=True)
        
        

        # For lla

        n.conv1_lla = L.Convolution(n.lla, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_lla = L.ReLU(n.conv1_lla, in_place=True)
        n.conv2_lla = L.Convolution(n.relu1_lla, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_lla = L.ReLU(n.conv2_lla, in_place=True)

        n.norm2_lla = L.LRN(n.relu2_lla, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_lla = L.Pooling(n.norm2_lla, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_lla = L.Convolution(n.pool2_lla, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_lla = L.ReLU(n.conv3_lla, in_place=True)
        '''
        n.conv4_lla = L.Convolution(n.relu3_lla, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_lla = L.ReLU(n.conv4_lla, in_place=True)
        '''
        if if_Train:
            n.drop4_lla = fc5input_lla  = L.Dropout(n.relu3_lla, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_lla = n.relu3_lla
        
        n.fc5_lla = L.InnerProduct(fc5input_lla, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_lla_relu = L.ReLU(n.fc5_lla, in_place=True)
        



        # For lshoe

        n.conv1_lshoe = L.Convolution(n.lshoe, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_lshoe = L.ReLU(n.conv1_lshoe, in_place=True)
        n.conv2_lshoe = L.Convolution(n.relu1_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_lshoe = L.ReLU(n.conv2_lshoe, in_place=True)

        n.norm2_lshoe = L.LRN(n.relu2_lshoe, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_lshoe = L.Pooling(n.norm2_lshoe, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_lshoe = L.Convolution(n.pool2_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_lshoe = L.ReLU(n.conv3_lshoe, in_place=True)
        '''
        n.conv4_lshoe = L.Convolution(n.relu3_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_lshoe = L.ReLU(n.conv4_lshoe, in_place=True)
        '''
        if if_Train:
            n.drop4_lshoe = fc5input_lshoe  = L.Dropout(n.relu3_lshoe, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_lshoe = n.relu3_lshoe
        
        n.fc5_lshoe = L.InnerProduct(fc5input_lshoe, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_lshoe_relu = L.ReLU(n.fc5_lshoe, in_place=True)
        
        

        # For rshoe

        n.conv1_rshoe = L.Convolution(n.rshoe, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_rshoe = L.ReLU(n.conv1_rshoe, in_place=True)
        n.conv2_rshoe = L.Convolution(n.relu1_rshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_rshoe = L.ReLU(n.conv2_rshoe, in_place=True)

        n.norm2_rshoe = L.LRN(n.relu2_rshoe, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_rshoe = L.Pooling(n.norm2_rshoe, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_rshoe = L.Convolution(n.pool2_rshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_rshoe = L.ReLU(n.conv3_rshoe, in_place=True)
        '''
        n.conv4_rshoe = L.Convolution(n.relu3_rshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_rshoe = L.ReLU(n.conv4_rshoe, in_place=True)
        '''
        if if_Train:
            n.drop4_rshoe = fc5input_rshoe  = L.Dropout(n.relu3_rshoe, dropout_ratio=0.2, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_rshoe = n.relu3_rshoe
        
        n.fc5_rshoe = L.InnerProduct(fc5input_rshoe, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_rshoe_relu = L.ReLU(n.fc5_rshoe, in_place=True)
        
        
        #Concatenation
        
        n.concat = L.Concat(*[n.fc5_acce_relu, n.fc5_back_relu, n.fc5_rua_relu, n.fc5_rla_relu,
                             n.fc5_lua_relu, n.fc5_lla_relu, n.fc5_lshoe_relu, n.fc5_rshoe_relu], concat_param = dict(axis = 1))
            
        if False:#if_Train:
            n.drop5 = fc6input  = L.Dropout(n.concat, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc6input = n.concat
    
        n.fc6 = L.InnerProduct(fc6input, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        
        if self.output == 'softmax':
        
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
            
        
        return n.to_proto()
    
    

    def network_cnn_imu_opportunity_3(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        
        n = caffe.NetSpec()
        

        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.acce, n.back, n.lshoe, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,36]), dict(dim=[batch_size,1,self.sliding_window_length,45]),
                                                                dict(dim=[batch_size,1,self.sliding_window_length,32]), dict(dim=[batch_size, 1])], ntop = 4)
        
        # For acce

        n.conv1_acce = L.Convolution(n.acce, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_acce = L.ReLU(n.conv1_acce, in_place=True)
        n.norm1_acce = L.LRN(n.relu1_acce, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2_acce = L.Convolution(n.norm1_acce, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_acce = L.ReLU(n.conv2_acce, in_place=True)

        n.norm2_acce = L.LRN(n.relu2_acce, local_size=5, alpha=1e-4, beta=0.75)
        n.pool2_acce = L.Pooling(n.norm2_acce, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_acce = L.Convolution(n.pool2_acce, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_acce = L.ReLU(n.conv3_acce, in_place=True)
        

        n.conv4_acce = L.Convolution(n.relu3_acce, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_acce = L.ReLU(n.conv4_acce, in_place=True)
        
        n.pool4_acce = L.Pooling(n.relu4_acce, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)

                
        if if_Train:
            n.drop4_acce = fc5input_acce  = L.Dropout(n.pool4_acce, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_acce = n.pool4_acce


        n.fc5_acce = L.InnerProduct(fc5input_acce, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_acce_relu = L.ReLU(n.fc5_acce, in_place=True)
        
        
        

        # For back

        n.conv1_back = L.Convolution(n.back, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_back = L.ReLU(n.conv1_back, in_place=True)
        n.norm1_back = L.LRN(n.relu1_back, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2_back = L.Convolution(n.norm1_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_back = L.ReLU(n.conv2_back, in_place=True)

        n.norm2_back = L.LRN(n.relu2_back, local_size=5, alpha=1e-4, beta=0.75)
        
        n.pool2_back = L.Pooling(n.norm2_back, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_back = L.Convolution(n.pool2_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_back = L.ReLU(n.conv3_back, in_place=True)
        
        n.conv4_back = L.Convolution(n.relu3_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_back = L.ReLU(n.conv4_back, in_place=True)
        
        n.pool4_back = L.Pooling(n.relu4_back, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)
        
        if if_Train:
            n.drop4_back = fc5input_back  = L.Dropout(n.pool4_back, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_back = n.pool4_back

        n.fc5_back = L.InnerProduct(fc5input_back, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_back_relu = L.ReLU(n.fc5_back, in_place=True)
             


        # For lshoe

        n.conv1_lshoe = L.Convolution(n.lshoe, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_lshoe = L.ReLU(n.conv1_lshoe, in_place=True)
        n.norm1_lshoe = L.LRN(n.relu1_lshoe, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2_lshoe = L.Convolution(n.norm1_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_lshoe = L.ReLU(n.conv2_lshoe, in_place=True)

        n.norm2_lshoe = L.LRN(n.relu2_lshoe, local_size=5, alpha=1e-4, beta=0.75)
        
        n.pool2_lshoe = L.Pooling(n.norm2_lshoe, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_lshoe = L.Convolution(n.pool2_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_lshoe = L.ReLU(n.conv3_lshoe, in_place=True)

        n.conv4_lshoe = L.Convolution(n.relu3_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_lshoe = L.ReLU(n.conv4_lshoe, in_place=True)
        
        n.pool4_lshoe = L.Pooling(n.relu4_lshoe, kernel_h = 2, kernel_w = 1, stride_h = 1, stride_w = 1, pool=P.Pooling.MAX)

        if if_Train:
            n.drop4_lshoe = fc5input_lshoe  = L.Dropout(n.pool4_lshoe, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_lshoe = n.pool4_lshoe
        
        n.fc5_lshoe = L.InnerProduct(fc5input_lshoe, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_lshoe_relu = L.ReLU(n.fc5_lshoe, in_place=True)
        
        
        
        
        #Concatenation
        
        n.concat = L.Concat(*[n.fc5_acce_relu, n.fc5_back_relu, n.fc5_lshoe_relu], concat_param = dict(axis = 1))
        
        #n.norm_concat = L.BatchNorm(n.concat)
        
        '''
        n.fc5 = L.InnerProduct(n.norm_concat, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        n.fc5_relu = L.ReLU(n.fc5, in_place=True)
        '''


        if use_maxout:
            if if_Train:
                n.droppremax = L.Dropout(n.concat, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size, int(512 * 3 / 2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.concat, shape=dict(dim=[batch_size,int(512 * 3 / 2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        else:
            
            if if_Train:
                n.drop5 = fc6input  = L.Dropout(n.concat, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
            else:
                fc6input = n.concat
        
            n.fc6 = L.InnerProduct(fc6input, num_output=256, param = self.learned_param, 
                                  weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        
        if self.output == 'softmax':
        
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
            
        
        return n.to_proto()
    
    

    def network_cnn_imu_opportunity_2(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        
        n = caffe.NetSpec()
        

        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.back, n.lshoe, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,81]),
                                                        dict(dim=[batch_size,1,self.sliding_window_length,32]),
                                                        dict(dim=[batch_size, 1])], ntop = 3)
               
        
        

        # For back

        n.conv1_back = L.Convolution(n.back, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_back = L.ReLU(n.conv1_back, in_place=True)
        n.norm1_back = L.LRN(n.relu1_back, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2_back = L.Convolution(n.norm1_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_back = L.ReLU(n.conv2_back, in_place=True)

        n.norm2_back = L.LRN(n.relu2_back, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_back = L.Pooling(n.norm2_back, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_back = L.Convolution(n.norm2_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_back = L.ReLU(n.conv3_back, in_place=True)
        
        n.conv4_back = L.Convolution(n.relu3_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_back = L.ReLU(n.conv4_back, in_place=True)
        
        if if_Train:
            n.drop4_back = fc5input_back  = L.Dropout(n.relu4_back, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_back = n.relu4_back

        n.fc5_back = L.InnerProduct(fc5input_back, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_back_relu = L.ReLU(n.fc5_back, in_place=True)
             


        # For lshoe

        n.conv1_lshoe = L.Convolution(n.lshoe, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_lshoe = L.ReLU(n.conv1_lshoe, in_place=True)
        n.norm1_lshoe = L.LRN(n.relu1_lshoe, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2_lshoe = L.Convolution(n.norm1_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_lshoe = L.ReLU(n.conv2_lshoe, in_place=True)

        n.norm2_lshoe = L.LRN(n.relu2_lshoe, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_lshoe = L.Pooling(n.norm2_lshoe, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_lshoe = L.Convolution(n.norm2_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_lshoe = L.ReLU(n.conv3_lshoe, in_place=True)

        n.conv4_lshoe = L.Convolution(n.relu3_lshoe, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_lshoe = L.ReLU(n.conv4_lshoe, in_place=True)

        if if_Train:
            n.drop4_lshoe = fc5input_lshoe  = L.Dropout(n.relu4_lshoe, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_lshoe = n.relu4_lshoe
        
        n.fc5_lshoe = L.InnerProduct(fc5input_lshoe, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_lshoe_relu = L.ReLU(n.fc5_lshoe, in_place=True)
        
        
        
        
        #Concatenation
        
        n.concat = L.Concat(*[n.fc5_back_relu, n.fc5_lshoe_relu], concat_param = dict(axis = 1))
        
        #n.norm_concat = L.BatchNorm(n.concat)
        
        '''
        n.fc5 = L.InnerProduct(n.norm_concat, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        n.fc5_relu = L.ReLU(n.fc5, in_place=True)
        '''

        if use_maxout:
            if if_Train:
                n.droppremax = L.Dropout(n.concat, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size, int(512 * 2/ 2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.concat, shape=dict(dim=[batch_size,int(512 * 2 / 2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        else:
            
            if if_Train:
                n.drop5 = fc6input  = L.Dropout(n.concat, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
            else:
                fc6input = n.concat
        
            n.fc6 = L.InnerProduct(fc6input, num_output=256, param = self.learned_param, 
                                  weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        
        if self.output == 'softmax':
        
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
            
        
        return n.to_proto()
    

    def network_cnn_imu_opportunity_1(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        
        n = caffe.NetSpec()
        

        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.back, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,113]),
                                                        dict(dim=[batch_size, 1])], ntop = 2)
               
        
        

        # For back

        n.conv1_back = L.Convolution(n.back, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_back = L.ReLU(n.conv1_back, in_place=True)
        n.norm1_back = L.LRN(n.relu1_back, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2_back = L.Convolution(n.norm1_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_back = L.ReLU(n.conv2_back, in_place=True)

        #n.norm2_back = L.LRN(n.relu2_back, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_back = L.Pooling(n.norm2_back, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        
        n.conv3_back = L.Convolution(n.relu2_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_back = L.ReLU(n.conv3_back, in_place=True)
        
        n.conv4_back = L.Convolution(n.relu3_back, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_back = L.ReLU(n.conv4_back, in_place=True)
        
        if if_Train:
            n.drop4_back = fc5input_back  = L.Dropout(n.relu4_back, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_back = n.relu4_back

        n.fc5_back = L.InnerProduct(fc5input_back, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_back_relu = L.ReLU(n.fc5_back, in_place=True)
                     
        
        
        #Concatenation
        
        #n.concat = L.Concat(*[n.fc5_back_relu], concat_param = dict(axis = 1))
        
        #n.norm_concat = L.BatchNorm(n.concat)
        
        '''
        n.fc5 = L.InnerProduct(n.norm_concat, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        n.fc5_relu = L.ReLU(n.fc5, in_place=True)
        '''

        if use_maxout:
            if if_Train:
                n.droppremax = L.Dropout(n.fc5_back_relu, dropout_ratio=0.5, in_place=False, include=dict(phase=caffe.TRAIN))
                n.premax = L.Reshape(n.droppremax, shape=dict(dim=[batch_size, int(512 / 2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            else:
                n.premax = L.Reshape(n.fc5_back_relu, shape=dict(dim=[batch_size,int(512 / 2),2,1]))
                n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
                n.fc6 = L.InnerProduct(n.maxout, num_output=256, param = self.learned_param, 
                      weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        else:
            
            if if_Train:
                n.drop5 = fc6input  = L.Dropout(n.fc5_back_relu, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
            else:
                fc6input = n.fc5_back_relu
        
            n.fc6 = L.InnerProduct(fc6input, num_output=256, param = self.learned_param, 
                                  weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        
        if self.output == 'softmax':
        
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
            
        
        return n.to_proto()


    

    def network_cnn_imu_pamap2(self, input_values, input_targets, batch_size = 1, if_Train = True, use_maxout = False):
        
        n = caffe.NetSpec()
        
        if self.fine_tunning:
            param = self.frozen_param
        else:
            param = self.learned_param
        
        if self.output == 'softmax':
            n.HR, n.Hand, n.Chest, n.Ankle, n.label = L.Input(shape = [dict(dim=[batch_size,1,self.sliding_window_length,1]),
                                                                       dict(dim=[batch_size,1,self.sliding_window_length,13]),
                                                                       dict(dim=[batch_size,1,self.sliding_window_length,13]),
                                                                       dict(dim=[batch_size,1,self.sliding_window_length,13]),
                                                                       dict(dim=[batch_size, 1])], ntop = 5)
        
        # For HR

        n.conv1_HR = L.Convolution(n.HR, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_HR = L.ReLU(n.conv1_HR, in_place=True)
        n.norm1_HR = L.LRN(n.relu1_HR, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool1_HR = L.Pooling(n.norm1_HR, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv2_HR = L.Convolution(n.norm1_HR, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_HR = L.ReLU(n.conv2_HR, in_place=True)
        n.norm2_HR = L.LRN(n.relu2_HR, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_HR = L.Pooling(n.norm2_HR, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv3_HR = L.Convolution(n.norm2_HR, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_HR = L.ReLU(n.conv3_HR, in_place=True)
        n.conv4_HR = L.Convolution(n.relu3_HR, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_HR = L.ReLU(n.conv4_HR, in_place=True)

        #n.pool4_HR = L.Pooling(n.relu4_HR, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)


        n.conv5_HR = L.Convolution(n.pool4_HR, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu5_HR = L.ReLU(n.conv5_HR, in_place=True)
        n.conv6_HR = L.Convolution(n.relu5_HR, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu6_HR = L.ReLU(n.conv6_HR, in_place=True)

        #n.pool6_HR = L.Pooling(n.relu6_HR, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)

                
        if if_Train:
            n.drop4_HR = fc5input_HR  = L.Dropout(n.relu6_HR, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_HR = n.relu6_HR


        n.fc5_HR = L.InnerProduct(fc5input_HR, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_HR_relu = L.ReLU(n.fc5_HR, in_place=True)
        
        
        

        # For Hand

        n.conv1_Hand = L.Convolution(n.Hand, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_Hand = L.ReLU(n.conv1_Hand, in_place=True)
        n.norm1_Hand = L.LRN(n.relu1_Hand, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool1_Hand = L.Pooling(n.norm1_Hand, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv2_Hand = L.Convolution(n.norm1_Hand, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_Hand = L.ReLU(n.conv2_Hand, in_place=True)
        n.norm2_Hand = L.LRN(n.relu2_Hand, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_Hand = L.Pooling(n.norm2_Hand, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv3_Hand = L.Convolution(n.norm2_Hand, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_Hand = L.ReLU(n.conv3_Hand, in_place=True)
        n.conv4_Hand = L.Convolution(n.relu3_Hand, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_Hand = L.ReLU(n.conv4_Hand, in_place=True)
        
        #n.pool4_Hand = L.Pooling(n.relu4_Hand, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)

        n.conv5_Hand = L.Convolution(n.pool4_Hand, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu5_Hand = L.ReLU(n.conv5_Hand, in_place=True)
        n.conv6_Hand = L.Convolution(n.relu5_Hand, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu6_Hand = L.ReLU(n.conv6_Hand, in_place=True)
        
        #n.pool6_Hand = L.Pooling(n.relu6_Hand, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)

        if if_Train:
            n.drop4_Hand = fc5input_Hand  = L.Dropout(n.relu6_Hand, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_Hand = n.relu6_Hand

        n.fc5_Hand = L.InnerProduct(fc5input_Hand, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_Hand_relu = L.ReLU(n.fc5_Hand, in_place=True)
        
        
        
        
        
        # For Chest

        n.conv1_Chest = L.Convolution(n.Chest, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_Chest = L.ReLU(n.conv1_Chest, in_place=True)
        n.norm1_Chest = L.LRN(n.relu1_Chest, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool1_Chest = L.Pooling(n.norm1_Chest, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv2_Chest = L.Convolution(n.norm1_Chest, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_Chest = L.ReLU(n.conv2_Chest, in_place=True)
        n.norm2_Chest = L.LRN(n.relu2_Chest, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_Chest = L.Pooling(n.norm2_Chest, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv3_Chest = L.Convolution(n.__Chest, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_Chest = L.ReLU(n.conv3_Chest, in_place=True)
        n.conv4_Chest = L.Convolution(n.relu3_Chest, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_Chest = L.ReLU(n.conv4_Chest, in_place=True)
        
        #n.pool4_Chest = L.Pooling(n.relu4_Chest, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)


        n.conv5_Chest = L.Convolution(n.pool4_Chest, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu5_Chest = L.ReLU(n.conv5_Chest, in_place=True)
        n.conv6_Chest = L.Convolution(n.relu5_Chest, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu6_Chest = L.ReLU(n.conv6_Chest, in_place=True)
        
        #n.pool6_Chest = L.Pooling(n.relu6_Chest, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)


        
        if if_Train:
            n.drop4_Chest = fc5input_Chest  = L.Dropout(n.relu6_Chest, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_Chest = n.relu6_Chest
        

        n.fc5_Chest = L.InnerProduct(fc5input_Chest, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_Chest_relu = L.ReLU(n.fc5_Chest, in_place=True)
        


        # For Ankle

        n.conv1_Ankle = L.Convolution(n.Ankle, kernel_size=[self.filter_size, 1],
                                stride_h = 1, stride_w=1, num_output=self.num_filters,
                                pad=0, param = param,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
        n.relu1_Ankle = L.ReLU(n.conv1_Ankle, in_place=True)
        n.norm1_Ankle = L.LRN(n.relu1_Ankle, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool1_Ankle = L.Pooling(n.norm1_Ankle, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv2_Ankle = L.Convolution(n.norm1_Ankle, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu2_Ankle = L.ReLU(n.conv2_Ankle, in_place=True)
        n.norm2_Ankle = L.LRN(n.relu2_Ankle, local_size=5, alpha=1e-4, beta=0.75)
        #n.pool2_Ankle = L.Pooling(n.norm2_Ankle, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)
        n.conv3_Ankle = L.Convolution(n.norm2_Ankle, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu3_Ankle = L.ReLU(n.conv3_Ankle, in_place=True)
        n.conv4_Ankle = L.Convolution(n.relu3_Ankle, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu4_Ankle = L.ReLU(n.conv4_Ankle, in_place=True)
        #n.pool4_Ankle = L.Pooling(n.relu4_Ankle, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)


        n.conv5_Ankle = L.Convolution(n.pool4_Ankle, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu5_Ankle = L.ReLU(n.conv5_Ankle, in_place=True)
        n.conv6_Ankle = L.Convolution(n.relu5_Ankle, kernel_size=[self.filter_size, 1], num_output=self.num_filters,
                                        pad=0, param = param,
                                        weight_filler=dict(type='msra'),
                                        bias_filler=dict(type='constant'))
        n.relu6_Ankle = L.ReLU(n.conv6_Ankle, in_place=True)
        #n.pool6_Ankle = L.Pooling(n.relu6_Ankle, kernel_h = 2, kernel_w = 1, stride_h = 2, stride_w = 1, pool=P.Pooling.MAX)

        
        if if_Train:
            n.drop4_Ankle = fc5input_Ankle  = L.Dropout(n.relu6_Ankle, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc5input_Ankle = n.relu6_Ankle
        
        n.fc5_Ankle = L.InnerProduct(fc5input_Ankle, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        
        n.fc5_Ankle_relu = L.ReLU(n.fc5_Ankle, in_place=True)
        
                
        
        
        #Concatenation
        
        n.concat = L.Concat(*[n.fc5_HR_relu, n.fc5_Hand_relu, n.fc5_Chest_relu, n.fc5_Ankle_relu], concat_param = dict(axis = 1))
            
        if if_Train:
            n.drop5 = fc6input  = L.Dropout(n.concat, dropout_ratio=0.5, in_place=True, include=dict(phase=caffe.TRAIN))
        else:
            fc6input = n.concat
    
        n.fc6 = L.InnerProduct(fc6input, num_output=512, param = self.learned_param, 
                              weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
                    
        

        n.fc6_relu = L.ReLU(n.fc6, in_place=True)
        
        
        if self.output == 'softmax':
        
            n.attrs = L.InnerProduct(n.fc6_relu, num_output=self.num_classes, param = self.learned_param, 
                                          weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
            
            n.loss = L.SoftmaxWithLoss(n.attrs, n.label)
            
            n.class_proba = L.Softmax(n.attrs, in_place=False)
            
        
            
        
        return n.to_proto()    



    def solver(self, train_dataset, target_values, niter, step_size, division_epochs = 1, batch_size = 1, if_Train = True, use_maxout = False):

        '''
        Returns the solver
    
        @param niter: Number of iterations
        @param train_dataset: Directory to the training dataset
        @param test_dataset: Directory to the testing dataset
        @param batch_size: Batch size for training
        @param num_classes: number of classes of the dataset
                            in the case of mnist, there are 10 clases
        @param learning_rate: Initial learning rate
        @param step_size: Drops the learning rate by a factor,
                            every step:size
        @param momentum: Value of momentum for the optimization
                        algorithm called SGD with momentum
                        
        
        '''    
        # @DELETE_START
        #Creating the training and testing networks
        
        if self.network_type == 'cnn_imu':
            if self.dataset == 'gesture' or self.dataset == 'locomotion':
                lenet_train = self.network_cnn_imu_opportunity_3(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, use_maxout = use_maxout)
                lenet_test = self.network_cnn_imu_opportunity_3(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, if_Train = False, use_maxout = use_maxout) 
            elif self.dataset == 'pamap2':
                lenet_train = self.network_cnn_imu_pamap2(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, use_maxout = use_maxout)
                lenet_test = self.network_cnn_imu_pamap2(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, if_Train = False, use_maxout = use_maxout) 
        elif self.network_type == 'cnn':
            if self.dataset == 'gesture' or self.dataset == 'locomotion':
                lenet_train = self.network_cnn(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, use_maxout = use_maxout)
                lenet_test = self.network_cnn(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, if_Train = False, use_maxout = use_maxout)
            elif self.dataset == 'pamap2':
                lenet_train = self.network_cnn_pamap2(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, use_maxout = use_maxout)
                lenet_test = self.network_cnn_pamap2(input_values =  train_dataset, input_targets = target_values, batch_size = batch_size, if_Train = False, use_maxout = use_maxout)
                
        
        # @DELETE_END
        
        cnn_architecture='cnn_based_network'
        if if_Train :
            train_net_proto_path = '../' + self.folder_exp + '/../prototxt/train_deepConv_' + self.dataset + '_train' + '.prototxt'
            test_net_proto_path = '../' + self.folder_exp + '/../prototxt/train_deepConv_' + self.dataset + '_test' + '.prototxt'
            solver_proto_path = '../' + self.folder_exp + '/../prototxt/train_deepConv_' + self.dataset + '_solver' + '.prototxt'
        elif if_Train == False:
            train_net_proto_path = '../' + self.folder_exp + '/../prototxt/test_deepConv_' + self.dataset + '_train' + '.prototxt'
            test_net_proto_path = '../' + self.folder_exp + '/../prototxt/test_deepConv_' + self.dataset + '_test' + '.prototxt'
            solver_proto_path = '../' + self.folder_exp + '/../prototxt/test_deepConv_' + self.dataset + '_solver' + '.prototxt'
            
        with open(train_net_proto_path, 'w') as proto_file:
                proto_file.write('#%s train \n' % (cnn_architecture))
                proto_file.write(str(lenet_train))
        
        with open(test_net_proto_path, 'w') as proto_file:
                proto_file.write('#%s test \n' % (cnn_architecture))
                proto_file.write(str(lenet_test))
                 
        solver_list = []
        solver_list.append('train_net: "%s"' % (train_net_proto_path))
        solver_list.append('test_net: "%s"' % (test_net_proto_path))
        solver_list.append('test_iter: %d' % (0))
        solver_list.append('test_interval: %d' % (niter * 2))


        if True:
            solver_list.append('base_lr: %f' % (self.lr))
            #solver_list.append('lr_policy: "inv"')
            solver_list.append('lr_policy: "step"')
            #solver_list.append('gamma: 0.0001')
            solver_list.append('gamma: 0.1')
            #solver_list.append('power: 0.75')
            solver_list.append('stepsize: %d' % (step_size))
        else:
        

            solver_list.append('lr_policy: "step_lr_pair"')
            
            step_sizes = range(0, niter, step_size)
            
            learning_rates = [self.lr]
            for dv in range(division_epochs - 2):
                learning_rates.append(self.lr *10)
            learning_rates.append(self.lr)

            
            for lr, ssize in zip(learning_rates, step_sizes):
                solver_list.append('stepvalue: %d' % (ssize))
                solver_list.append('lrvalue: %.10f' % (lr))
        
        
        solver_list.append('display: %d' % (niter * 2))
        solver_list.append('iter_size: %d' % (1))
        solver_list.append('max_iter: %d' % (niter))
        #solver_list.append('momentum: %f' % (0.9)) #0.95 for SGD
        solver_list.append('weight_decay: 0.00000')
        solver_list.append('snapshot: 0')
        solver_list.append('solver_mode: GPU')
        solver_list.append('type: "%s"' % ("RMSProp"))
        #solver_list.append('type: "%s"' % ("SGD"))
        solver_list.append('rms_decay: %f' % (0.95))
        
        self.write_list(file_path=solver_proto_path, line_list=solver_list)
        
        
        solver = caffe.get_solver(solver_proto_path)
        
        return solver
    
    

    def write_list(self, file_path, line_list, encoding='ascii'):
        '''
        Writes a list into the given file object
        
        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''        
        with io.open(file_path, 'w', encoding=encoding) as f:
            for l in line_list:
                f.write(unicode(l) + '\n')
        
