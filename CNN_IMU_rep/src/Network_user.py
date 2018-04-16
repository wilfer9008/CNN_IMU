'''
Created on Feb 28, 2017

@author: fmoya
'''


import sys
import numpy as np
import os

import matplotlib.pyplot as plt 

import cPickle as cp

from sliding_window import sliding_window

import sklearn.metrics as metrics

from scipy.spatial.distance import cdist

from numpy import linalg as LA

import caffe
from caffe import layers as L
from caffe import params as P



import lasagne

from CNN import CNN



class Network_user(object):
    '''
    classdocs
    '''


    def __init__(self, folder_exp, config, folder_exp_fine_tuning = None, folder_exp_test = None):
        '''
        Constructor
        '''
        
        self.logger = None
        self.sliding_window_length = config['sliding_window_length']
        self.NB_sensor_channels = config['NB_sensor_channels']
        self.num_filters = config['num_filters']
        self.filter_size = config['filter_size']
        self.lr = config['lr']
        self.epochs = config['epochs'] 
        self.num_classes = config['num_classes']
        self.train_show = config['train_show']
        self.valid_show = config['valid_show']
        self.sliding_window_step =  config['sliding_window_step']
        self.folder_exp = folder_exp
        self.current_net = None
        self.plotting = config['plotting']
        self.network = config['network']
        self.usage_modus = config['usage_modus']
        self.dataset = config['dataset']
        self.output = config['output']
        self.fine_tunning = config['fine_tunning']
        self.use_maxout = config['use_maxout']
        self.model_fine_tuning = config['model_fine_tuning']
        self.folder_exp_fine_tuning = folder_exp_fine_tuning
        self.balancing = config['balancing']
        self.division_epochs = config['division_epochs']
        self.GPU = config['GPU']
        self.folder_exp_test = folder_exp_test
        self.num_block = config['num_block']
        
        
        
        return
    

    def set_logger(self, logger):
        
        self.logger = logger
        
        self.logger.info("User: Setting logger in folder {}...".format(self.folder_exp))
        print("User: Setting logger in folder {}...".format(self.folder_exp))
        
        return
    

    def load_dataset(self, filename):
        
        self.logger.info("CNN: loading dataset")
        print "CNN: loading dataset"
        
        f = file(filename, 'rb')
        data = cp.load(f)
        f.close()
    
        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]
    
    
        self.logger.info(" ..from file {}".format(filename))
        print(" ..from file {}".format(filename))
        self.logger.info(" ..reading instances: train {0}, val {1}, test {2}".format(X_train.shape, X_val.shape, X_test.shape))
        print(" ..reading instances: train {0}, val {1}, test {2}".format(X_train.shape, X_val.shape, X_test.shape))
    
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
    
        # The targets are casted to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_val = y_val.astype(np.uint8)
        y_test = y_test.astype(np.uint8)
    
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    
    def set_epochs(self, new_epochs):
        
        self.epochs = new_epochs
        return
    
    
    def set_num_filters(self, num_filters):
    
        self.num_filters = num_filters
        return 
    
    
    
    
    def opp_sliding_window(self, data_x, data_y, ws, ss, label_pos_end = True):

        self.logger.info("Sliding window: Creating windows {} with step {}".format(ws, ss))
        print("Sliding window: Creating windows {} with step {}".format(ws, ss))
        
        data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
        if label_pos_end:
            data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
        else:
            
            if False:
                data_y_labels = np.asarray([[i[i.shape[0] // 2]] for i in sliding_window(data_y,ws,ss)])
            else:
                
                try:
                    data_y_labels = []
                    for sw in sliding_window(data_y,ws,ss):
                        count_l = np.bincount(sw, minlength = self.num_classes)
                        idy = np.argmax(count_l)
                        data_y_labels.append(idy)
                    data_y_labels = np.asarray(data_y_labels)
                         
        
                except:
                    self.logger.info("Sliding window: error with the counting {}".format(count_l))
                    print("Sliding window: error with the counting {}".format(count_l))
                    self.logger.info("Sliding window: error with the counting {}".format(idy))
                    print("Sliding window: error with the counting {}".format(idy))
                    return np.Inf
   
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y,ws,ss)])
        return data_x.astype(np.float32), data_y_labels.reshape(len(data_y_labels)).astype(np.uint8), data_y_all.astype(np.uint8)
    

    def create_batches(self, data, batch_size = 1):
    
        self.logger.info("Prepare: Preparing data with batch size {}".format(batch_size))
        print "Prepare: Preparing data with batch size {}".format(batch_size)
        data_batches = []
        batches = np.arange(0, data.shape[0], batch_size)
        
        for idx in range(batches.shape[0] - 1 ):
            batch = []
            for data_in_batch in data[batches[idx]: batches[idx + 1]]:
                channel = []
                channel.append(data_in_batch.astype(np.float32))
                batch.append(channel)
            data_batches.append(batch)
            
        data_batches = np.array(data_batches)
        return data_batches
    

    def random_data(self, data, label, y_data = None):
        if data.shape[0] != label.shape[0]:
            self.logger.info("Random: Data and label havent the same number of samples")
            print "Random: Data and label havent the same number of samples"
            raise RuntimeError('Random: Data and label havent the same number of samples')
        
        if False:#os.path.isfile('../' + self.folder_exp + '/random_train_order.pkl'):
            self.logger.info("Random: Getting random order")
            print "Random: Getting random order"
            
            file2idx = cp.load(open('../' + self.folder_exp + '/random_train_order.pkl'))
            idx = file2idx["idx"]
            
        else:
            idx = np.arange(data.shape[0])
            np.random.shuffle(idx)
            
            idx2file = {"idx" : idx}
            f = open('../' + self.folder_exp + '/random_train_order.pkl', 'wb')
            cp.dump(idx2file, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
        
        data_s = data[idx]
        label_s = label[idx]
        
        if y_data is not None:
            y_data_s = y_data[idx]
        else:
            y_data_s = y_data
        
        return data_s, label_s, y_data_s
    
    
    def prepare_data(self, val, lab, if_val = False, batch_size = 1, y_data = None):    
        if if_val == False:
            train_vals_fl, train_labels_fl, y_data_fl = self.random_data(val, lab, y_data = y_data)
        else:
            train_vals_fl = val
            train_labels_fl = lab
            y_data_fl = y_data
        
        self.logger.info("Data: Creating batches...")
        print "Data: Creating batches..."
        
        v_b = np.array(self.create_batches(np.array(train_vals_fl), batch_size = batch_size))
        l_b = np.array(self.create_batches(np.array(train_labels_fl), batch_size = batch_size))
        
        if y_data is not None:
            y_data_b = np.array(self.create_batches(np.array(y_data_fl), batch_size = batch_size))
        else:
            y_data_b = None
        
        return v_b.astype(np.float32), l_b.astype(np.float32), y_data_b
    
    
    
    def balanced_epoch_dataset(self, X_train_in, y_train_in, statistics, batch_size = 1, y_data_in = None):
    
        
        train_short_data_v = []
        train_short_data_l = []
        train_short_y_data = []
        for ke in statistics.keys():
            where_ke = np.where(y_train_in == ke)
            where_ke = where_ke[0]
            np.random.shuffle(where_ke)
            
            #print "Train: Checking idx {} with #{} samples [] random {}".format(ke, statistics[ke],where_ke[0:5])
            
            idx_ke = where_ke[:np.min(statistics.values())+int( (statistics[ke]-np.min(statistics.values())) * 0.75)]
            for elemt in idx_ke:
                if y_train_in[elemt] == ke:
                    train_short_data_v.append(X_train_in[elemt])
                    train_short_data_l.append(y_train_in[elemt])
                    train_short_y_data.append(y_data_in[elemt])
                else:
                    print "Wrong balanced layer"
                
        
        train_short_data_v = np.array(train_short_data_v)
        train_short_data_l = np.array(train_short_data_l)
        train_short_y_data = np.array(train_short_y_data)
        
        #for l_t in range(self.num_classes):
        #    print "Number of samples {} in label {}".format( np.sum(train_short_data_l == l_t), l_t) 
            
        X_train, y_train, y_data = self.prepare_data(train_short_data_v, train_short_data_l, batch_size = batch_size, y_data = train_short_y_data)
        
        #for l_t in range(self.num_classes):
        #    print "In y_train Number of samples {} in label {}".format( np.sum(y_train == l_t), l_t) 
        
        
        return X_train, y_train, y_data
    
    
    
    
    def set_orthonormal_conv(self, n,c,w,h, gain):
        '''
        Orthonornal initialization for convolutional layer
        @param n: number of batch
        @param c: number of channels
        @param w: number of weight
        @param h: number of height
        @param gain: 1 for sigmoid, sqrt(2) for relu
        
        '''
        
        float_shape = (n, c * w * h)
        X = np.random.random(float_shape)
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
    
        Q = U if U.shape == float_shape else Vt
        self.logger.info("CNN: Ortonormal Vt {}\n".format(Vt.shape))
        print "CNN: Ortonormal Vt {}\n".format(Vt.shape)
        np.allclose(np.dot(Q, Q.T), np.eye(Q.shape[0]))
        W = Q.reshape((n, c, w, h))
        
        return W * gain
    
    def set_orthonormal_fc(self, n,c, gain):
        '''
        Orthonornal initialization for fully connected layer
        @param n: number of batch
        @param c: number of channels
        @param w: number of weight
        @param h: number of height
        @param gain: 1 for sigmoid, sqrt(2) for relu
        
        '''
        
        float_shape = (n, c)
        X = np.random.random(float_shape)
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
    
        Q = U if U.shape == float_shape else Vt
        self.logger.info("CNN: Ortonormal Q {}\n".format(Q.shape))
        print "CNN: Ortonormal Q {}\n".format(Q.shape)
        np.allclose(np.dot(Q, Q.T), np.eye(Q.shape[0]))
        W = Q.reshape((n, c))
        
        return W * gain
    
    
    def set_weights(self):
        
        weight_layer_names = self.current_net.net.params.keys()
        weight_layers = {}
        
        for layer in weight_layer_names:
            if layer != 'conv1' and layer != 'conv1_HR' and layer != 'conv1_Hand' and layer != 'conv1_Chest' and layer != 'conv1_Ankle' :
                weight_layers[layer] = self.current_net.net.params[layer][0].data[...].copy()
                self.logger.info("CNN: Layer's names {}".format(layer))
                print "CNN: Layer's names {}".format(layer)
        
        w_new = {}
        for layer in weight_layers.keys():
            self.logger.info("CNN: obtanining weights network {}".format(layer))
            print "CNN: obtanining weights network {}".format(layer)

            w = weight_layers[layer]
            gain = 1
            if len(w.shape) == 4:
                w_new[layer] = self.set_orthonormal_conv(w.shape[0],w.shape[1],w.shape[2],w.shape[3], gain)
            elif len(w.shape) == 2:
                w_new[layer] = self.set_orthonormal_fc(w.shape[0],w.shape[1], gain)
                
            
        for layer in w_new.keys() :
            weight_new = w_new[layer]
            self.logger.info("CNN: Setting weights network in layer {} with {} weights".format(layer, str(weight_new.shape)))
            print "CNN: Setting weights network in layer {} with {} weights".format(layer, str(weight_new.shape))
            
            self.current_net.net.params[layer][0].data[...] = weight_new
            
            
        return
    
    
    
    def save_network(self, itera, name_net = 'best_deepConv_weights'):
        
        if self.network == 'cnn' or self.network == 'cnn_imu':
            self.logger.info("Train: Saving the weights \n")
            print "Train: Saving the weights \n"
            
            weights = os.path.join('../' + self.folder_exp + '/' + name_net + '.caffemodel')
            self.current_net.net.save(weights) 
        
        elif self.network == 'lstm':

            network_dict = {'params' : lasagne.layers.get_all_param_values(self.current_net['output'])}
            f = open('../' + self.folder_exp + '/' + name_net + '.pkl', 'wb')
            cp.dump(network_dict, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
        
        return
        
    
    
    def copy_weights(self, model, weights):
        
        
        VGG_net = caffe.Net(model, weights, caffe.TEST)
        weight_layer_names = self.current_net.net.params.keys()
        weight_layers = {}
        bias_layers = {}
        
        if self.network == 'cnn':
            for layer in weight_layer_names:
                self.logger.info("CNN: Layer's names {}".format(layer))
                print "CNN: Layer's names {}".format(layer)
                if layer == 'conv1':
                    weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                    bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                    self.logger.info("CNN: copying layer {}".format(layer))
                    print("CNN: copying layer {}".format(layer))
                if layer == 'conv2':
                    weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                    bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                    self.logger.info("CNN: copying layer {}".format(layer))
                    print("CNN: copying layer {}".format(layer))

                if layer == 'conv3':
                    weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                    bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                    self.logger.info("CNN: copying layer {}".format(layer))
                    print("CNN: copying layer {}".format(layer))
                if layer == 'conv4':
                    weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                    bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                    self.logger.info("CNN: copying layer {}".format(layer))
                    print("CNN: copying layer {}".format(layer))

                    
            for layer in weight_layers.keys():
                self.current_net.net.params[layer][0].data[...] = weight_layers[layer].copy()
                self.current_net.net.params[layer][1].data[...] = bias_layers[layer].copy()
                

        if self.network == 'cnn_imu':
            if self.dataset == 'locomotion' or self.dataset == 'gesture':
                for layer in weight_layer_names:
                    self.logger.info("CNN: Layer's names {}".format(layer))
                    print "CNN: Layer's names {}".format(layer)
                    if layer == 'conv1_acce':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
                    if layer == 'conv2_acce':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
    
                    if layer == 'conv3_acce':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
                    if layer == 'conv4_acce':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
    
                    if layer == 'conv1_back':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
                    if layer == 'conv2_back':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
    
                    if layer == 'conv3_back':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
                    if layer == 'conv4_back':
                        weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                        bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                        self.logger.info("CNN: copying layer {}".format(layer))
                        print("CNN: copying layer {}".format(layer))
    
                        
                for layer in weight_layers.keys():
                    self.current_net.net.params[layer][0].data[...] = weight_layers[layer].copy()
                    self.current_net.net.params[layer][1].data[...] = bias_layers[layer].copy()
                
        del VGG_net
        del weight_layers
        del bias_layers
        
        return
    

    def copy_weights_best(self, model, weights):
        
        
        VGG_net = caffe.Net(model, weights, caffe.TEST)
        weight_layer_names = self.current_net.net.params.keys()
        weight_layers = {}
        bias_layers = {}
        
        if self.network == 'cnn':
            for layer in weight_layer_names:
                self.logger.info("CNN: Layer's names {}".format(layer))
                print "CNN: Layer's names {}".format(layer)
                weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                self.logger.info("CNN: copying layer {}".format(layer))
                print("CNN: copying layer {}".format(layer))

                    
            for layer in weight_layers.keys():
                self.current_net.net.params[layer][0].data[...] = weight_layers[layer].copy()
                self.current_net.net.params[layer][1].data[...] = bias_layers[layer].copy()
                

        if self.network == 'cnn_imu':
            if self.dataset == 'locomotion' or self.dataset == 'gesture':
                for layer in weight_layer_names:
                    self.logger.info("CNN: Layer's names {}".format(layer))
                    print "CNN: Layer's names {}".format(layer)
                    weight_layers[layer] = VGG_net.params[layer][0].data[...].copy()
                    bias_layers[layer] = VGG_net.params[layer][1].data[...].copy()
                    self.logger.info("CNN: copying layer {}".format(layer))
                    print("CNN: copying layer {}".format(layer))
    
    
                        
                for layer in weight_layers.keys():
                    self.current_net.net.params[layer][0].data[...] = weight_layers[layer].copy()
                    self.current_net.net.params[layer][1].data[...] = bias_layers[layer].copy()
                
        del VGG_net
        del weight_layers
        del bias_layers
        
        return
    
    
    def get_precision_recall(self, targets, predictions):
        
        precision = np.zeros((self.num_classes))
        recall = np.zeros((self.num_classes))
        for c in range(self.num_classes):
            selected_elements = np.where(predictions == c)[0]
            non_selected_elements = np.where(predictions != c)[0]
            
            true_positives = np.sum(targets[selected_elements] == c)
            false_positives = np.sum(targets[selected_elements] != c)
            
            false_negatives = np.sum(targets[non_selected_elements] == c)
            
            precision[c] = true_positives / float(true_positives + false_positives)
            recall[c] = true_positives / float(true_positives + false_negatives)
        
        
        return precision, recall
    
    def f1_metric(self, targets, predictions, type = 'weighted'):
        
        precision, recall = self.get_precision_recall(targets, predictions)
        
        proportions = np.zeros((self.num_classes))
        if type == 'weighted':
            for c in range(self.num_classes):
                proportions[c] = np.sum(targets == c) / float(targets.shape[0])
            mult_pre_rec = precision * recall
            sum_pre_rec = precision + recall
    
            mult_pre_rec[np.isnan(mult_pre_rec)] = 0
            sum_pre_rec[np.isnan(sum_pre_rec)] = 0
            
            weighted_f1 = proportions * (mult_pre_rec / sum_pre_rec) 
            
            weighted_f1[np.isnan(weighted_f1)] = 0
            
            F1 = np.sum(weighted_f1) * 2
                
            print "Weighted F1 {}".format(F1)
            
        elif type == 'mean':
            mult_pre_rec = precision * recall
            sum_pre_rec = precision + recall
    
            mult_pre_rec[np.isnan(mult_pre_rec)] = 0
            sum_pre_rec[np.isnan(sum_pre_rec)] = 0
            
            f1 = mult_pre_rec / sum_pre_rec
            f1[np.isnan(f1)] = 0
            
            F1 = np.sum(f1) * 2 / self.num_classes
            
            print "Mean F1 {}".format(F1)
        
        return F1
        
        
        
    
    def train(self, X_train_in, y_train_in, X_val_in, y_val_in, statistics, ea_itera, batch_size = 1, y_data_in = None):
        
        self.logger.info("EA_iter {} Training network...".format(ea_itera))
        print "EA_iter {} Training network...".format(ea_itera)
        
        return
    
    
    
    
            


    def test_dist(self, X_test_in, y_test_in, ea_itera, in_train = True):
        
        #self.logger.info()
        
        self.logger.info("EA_iter {} Testing network...".format(ea_itera))
        print "EA_iter {} Testing network...".format(ea_itera)
            
            
        
        self.logger.info("Test: Loading weights")
        print "Test: Loading weights"
        
        try:
            if self.network == 'cnn' or self.network == 'cnn_imu':
                if self.num_block == 1:
                    if in_train == True:
                        if os.path.isfile('../' + self.folder_exp + '/../caffemodel/deepConv_weights_' + self.network +'.caffemodel'):
                            weights = '../' + self.folder_exp + '/../caffemodel/deepConv_weights_' + self.network +'.caffemodel'
                            self.logger.info('Caffe VGG weights found')
                            print 'Caffe VGG weights found'
                        net_test = caffe.Net('../' + self.folder_exp + '/../prototxt/'+ self.network  + '_' + self.dataset + '_test.prototxt', weights, caffe.TEST)  
                    
                    
                else:
                    if in_train == True:
                        if os.path.isfile('../' + self.folder_exp + '/../caffemodel/deepConv_weights_' + self.network +'2.caffemodel'):
                            weights = '../' + self.folder_exp + '/../caffemodel/deepConv_weights_' + self.network +'2.caffemodel'
                            self.logger.info('Caffe VGG weights found')
                            print 'Caffe VGG weights found'
                        net_test = caffe.Net('../' + self.folder_exp + '/../prototxt/'+ self.network  + '2_' + self.dataset + '_test.prototxt', weights, caffe.TEST)   
                    

                print 'weights copied'
                
                
                
        except:
            self.logger.info("EA_iter {} network file does not exist".format(ea_itera))
            print "EA_iter {} network file does not exist".format(ea_itera)
            return np.Inf
                
        X_test, y_test, _  = self.prepare_data(X_test_in, y_test_in, if_val = True, batch_size = 1)
        
        self.logger.info(" ..after creating batches (validation): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
        print(" ..after creating batches (validation): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
        
            
        f, ax_v = plt.subplots(2, sharex=True)
        line1_v, = ax_v[0].plot([], [],'-r',label='red')
        line2_v, = ax_v[1].plot([], [],'-b',label='blue')
        
        costs_test = []
        accs_test = []
        
        

        cost_test = 0
        acc_test = 0

        
        iterations_t = X_test.shape[0]
        if self.output == 'softmax':
            deep_descriptor = np.empty((0,self.num_classes))
        
        predictions_labels = []
        for b in range(iterations_t):
            if b % 100 == 0:
                sys.stdout.write('\r' + '            Testing batch {} from {}'.format(str(b),str(X_test.shape[0])))
                sys.stdout.flush()
  
                        
            test_batch_v = X_test[b]
            test_batch_l = y_test[b]                
                


            if self.network == 'cnn':
                net_test.blobs['data'].data[...] = test_batch_v
                net_test.blobs['label'].data[...] = test_batch_l.astype(int)

                
            elif self.network == 'cnn_imu':
                if self.dataset == 'locomotion' or self.dataset == 'gesture':

                    net_test.blobs['acce'].data[...] = test_batch_v[:,:,:,0:36]
                    net_test.blobs['back'].data[...] = test_batch_v[:,:,:,36:81]
                    net_test.blobs['lshoe'].data[...] = test_batch_v[:,:,:,81:113]
                    
                    
                elif self.dataset == 'pamap2':
                    
                    
                    net_test.blobs['HR'].data[...] = np.reshape(test_batch_v[:,:,:,0],
                                                                        newshape = (test_batch_v[:,:,:,0].shape[0],
                                                                                    test_batch_v[:,:,:,0].shape[1],
                                                                                    test_batch_v[:,:,:,0].shape[2], 1))
                    net_test.blobs['Hand'].data[...] = test_batch_v[:,:,:,1:14]
                    net_test.blobs['Chest'].data[...] = test_batch_v[:,:,:,14:27]
                    net_test.blobs['Ankle'].data[...] = test_batch_v[:,:,:,27:40]
                    net_test.blobs['label'].data[...] = test_batch_l.astype(int)
                    
                    
                '''
                else:
                    net.test_nets[0].blobs['acce'].data[...] = test_batch_v[:,:,:,0:36]
                    net.test_nets[0].blobs['back'].data[...] = test_batch_v[:,:,:,36:45]
                    net.test_nets[0].blobs['rua'].data[...] = test_batch_v[:,:,:,45:54]
                    net.test_nets[0].blobs['rla'].data[...] = test_batch_v[:,:,:,54:63]
                    net.test_nets[0].blobs['lua'].data[...] = test_batch_v[:,:,:,63:72]
                    net.test_nets[0].blobs['lla'].data[...] = test_batch_v[:,:,:,72:81]
                    net.test_nets[0].blobs['lshoe'].data[...] = test_batch_v[:,:,:,81:97]
                    net.test_nets[0].blobs['rshoe'].data[...] = test_batch_v[:,:,:,97:113]'''


            net_test.forward()

            
            cost_test += net_test.blobs['loss'].data.copy()
            atts_batch_val = net_test.blobs['class_proba'].data.copy()


                
            prediction = np.argmax(atts_batch_val, axis = 1)
            acc_test += np.sum(prediction == test_batch_l.flatten().astype(int))            
            predictions_labels.append(np.argmax(atts_batch_val).astype(int))
            
            deep_descriptor = np.append(deep_descriptor, atts_batch_val, axis = 0)
                    
                        
                
            
            predictions_labels_2_show = np.array(predictions_labels)
            costs_test.append(cost_test / np.float(b+1))
            accs_test.append(acc_test / np.float(b+1))
            
            if (b+1) % 500 == 0:
                
                self.logger.info("EA_iter {} \n".format(ea_itera))
                print "EA_iter {} \n".format(ea_itera)
                
                if self.output == 'softmax':
                    self.logger.info("    Test: iter {} \n Pred {} \n Tgt {}".format(b, predictions_labels_2_show[0], test_batch_l[0,:]))
                    print "    Test: iter {} \n Pred {} \n Tgt {}".format(b, predictions_labels_2_show[0], test_batch_l[0,:])
                elif self.output == 'attribute':
                    self.logger.info("    Test: iter {} \n Pred {} \n Tgt {}".format(b, predictions_labels_2_show[0,:], test_batch_l[0,:]))
                    print "    Test: iter {} \n Pred {} \n Tgt {}".format(b, predictions_labels_2_show[0,:], test_batch_l[0,:])
                
                self.logger.info("    Test: iter {} cost {}".format(b, cost_test / np.float(b+1)))
                print "    Test: iter {} cost {}".format(b, cost_test / np.float(b+1))
                
                self.logger.info("    Test: iter {} acc {} \n".format(b, acc_test / np.float(b+1)) )
                print "    Test: iter {} acc {} \n".format(b, acc_test / np.float(b+1))  
                                                                      
        
                line1_v.set_ydata(costs_test)
                line1_v.set_xdata(range(len(costs_test)))
                line2_v.set_ydata(accs_test)
                line2_v.set_xdata(range(len(accs_test)))
                
                ax_v[0].relim()
                ax_v[0].autoscale_view()
                ax_v[1].relim()
                ax_v[1].autoscale_view()
                
                plt.draw() 
                plt.pause(0.05)     
            
        predictions_labels = np.array(predictions_labels)
        predictions_labels = predictions_labels.flatten()
        acc_f1 = metrics.f1_score(np.reshape(y_test, newshape = y_test.shape[0]), predictions_labels, average='weighted')    
        acc_fm = metrics.fbeta_score(np.reshape(y_test, newshape = y_test.shape[0]), predictions_labels, beta = 1, average= 'macro')
               
        
        
        self.logger.info("    Test ACC: acc in iters {} f1 {} and fm {}\n".format(acc_test / float(iterations_t), acc_f1, acc_fm) )
        print("    Test ACC: acc in iters {} f1 {} and fm {}\n".format(acc_test / float(iterations_t), acc_f1, acc_fm) )
        
        #Final Accuracy
        f_acc = np.sum( np.reshape(y_test, newshape = y_test.shape[0]) == predictions_labels ) / float(predictions_labels.shape[0])
        

        self.logger.info("    Test ACC: final acc in iters {} f1 {} and fm {}\n".format(f_acc, acc_f1, acc_fm) )
        print("    Test ACC: final acc in iters {} f1 {} and fm {}\n".format(f_acc, acc_f1, acc_fm) )
        
        #Using made metrics
        #precision, recall = self.get_precision_recall(targets = np.reshape(y_test, newshape = y_test.shape[0]), predictions = predictions_labels)
        
        self.f1_metric(targets = np.reshape(y_test, newshape = y_test.shape[0]), predictions = predictions_labels, type = 'weighted')
        self.f1_metric(targets = np.reshape(y_test, newshape = y_test.shape[0]), predictions = predictions_labels, type = 'mean')
        
        if self.usage_modus == 'test':
            #Saving the predictions
            test_data = {'predictions_labels' : predictions_labels, 'targets_labels' : np.reshape(y_test, newshape = y_test.shape[0]),
                         'deep_descriptor' : deep_descriptor}
            
            file_test = open('../' + self.folder_exp + '/predictions_labels.pkl', 'wb')
            cp.dump(test_data, file_test, protocol=cp.HIGHEST_PROTOCOL)
            file_test.close()
            
                    
        plt.close()
        
        # Computing confusion matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
        for cl in range(self.num_classes):
            pos_pred = predictions_labels == cl
            pos_pred_trg = np.reshape(y_test, newshape = y_test.shape[0])[pos_pred]
            bincount = np.bincount(pos_pred_trg.astype(int), minlength = self.num_classes)
            confusion_matrix[cl,:] = bincount
        
        
        self.logger.info("    Test ACC: Confusion matrix \n{}\n".format(confusion_matrix.astype(int)))
        print "    Test ACC: Confusion matrix \n{}\n".format(confusion_matrix.astype(int))
        
        percentage_pred = []
        for cl in range(self.num_classes):
            pos_trg = np.reshape(y_test, newshape = y_test.shape[0]) == cl
            percentage_pred.append(confusion_matrix[cl,cl] / float(np.sum(pos_trg)))
        percentage_pred = np.array(percentage_pred)
        
        
        self.logger.info("    Test ACC: Percentage PRedictions \n{}\n".format(percentage_pred))
        print "    Test ACC: Percentage PRedictions \n{}\n".format(percentage_pred)
        
        
        del X_test
        del net_test
        del costs_test
        del accs_test
        del predictions_labels_2_show
        
        return acc_test / float(iterations_t), acc_f1, percentage_pred, deep_descriptor, y_test       
          
       
    
    def evaluating_attr(self, ea_itera ,batch_size = 1, train_test_modus = 0):
        
        if self.network == 'cnn' or self.network == 'cnn_imu':
            self.logger.info("Setting GPU in caffe {} EA_iter {} used folder {}...".format(2, ea_itera, self.folder_exp))
            print("Setting GPU in caffe {} EA_iter {} used folder {}...".format(2, ea_itera, self.folder_exp))
            caffe.set_mode_gpu()
            caffe.set_device(self.GPU)
        
        
        
        self.logger.info("EA_iter {} used folder {}...".format(ea_itera, self.folder_exp))
        print("EA_iter {} used folder {}...".format(ea_itera, self.folder_exp))
        
        self.logger.info("EA_iter {} Loading data...".format(ea_itera))
        print("EA_iter {} Loading data...".format(ea_itera))
        
        
        if self.dataset == 'locomotion':
            X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset('/data/fmoya/HAR/opportunity/train_val_test_dataset_locomotion.data')
        elif self.dataset == 'gesture':
            X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset('/data/fmoya/HAR/opportunity/train_val_test_dataset_2.data')
        elif self.dataset == 'pamap2':
            X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset('/data/fmoya/HAR/pamap2/train_val_test_dataset_pamap2_12_classes_norm.data')
        if train_test_modus == 0:
            del X_test
            del y_test
        
        self.logger.info("EA_iter {} Data loaded".format(ea_itera))
        print "EA_iter {} Data loaded".format(ea_itera)
    
        assert self.NB_sensor_channels == X_train.shape[1]
        
        
        # Sensor data is segmented using a sliding window mechanism
        X_train, y_train, y_train_data = self.opp_sliding_window(X_train, y_train, self.sliding_window_length, self.sliding_window_step, label_pos_end = False)
        X_val, y_val, y_val_data = self.opp_sliding_window(X_val, y_val, self.sliding_window_length, self.sliding_window_step, label_pos_end = False)
        
        self.logger.info("EA_iter {}  ..after sliding window (training): inputs {}, targets {}".format(ea_itera, X_train.shape, y_train.shape))
        print("EA_iter {}  ..after sliding window (training): inputs {}, targets {}".format(ea_itera, X_train.shape, y_train.shape))
        self.logger.info("EA_iter {}  ..after sliding window (validation): inputs {}, targets {}".format(ea_itera, X_val.shape, y_val.shape))
        print("EA_iter {}  ..after sliding window (validation): inputs {}, targets {}".format(ea_itera, X_val.shape, y_val.shape))
        
        if train_test_modus != 0:
            X_test, y_test, y_test_data = self.opp_sliding_window(X_test, y_test, self.sliding_window_length, self.sliding_window_step, label_pos_end = False)
            self.logger.info("EA_iter {}  ..after sliding window (testing): inputs {}, targets {}".format(ea_itera, X_test.shape, y_test.shape))
            print("EA_iter {}  ..after sliding window (testing): inputs {}, targets {}".format(ea_itera, X_test.shape, y_test.shape))
            del y_test_data
        

            
            
            
        if train_test_modus == 0:

            self.logger.info("EA_iter {} Training and validating ...".format(ea_itera))
            print("EA_iter {} Training and validating ...".format(ea_itera))
                        
            
            
        
        elif train_test_modus == 1:

            self.logger.info("EA_iter {} Final training and testing...".format(ea_itera))
            print("EA_iter {} Final training and testing...".format(ea_itera))

            
        else:            


            self.logger.info("EA_iter {} Final testing...".format(ea_itera))
            print("EA_iter {} Final training and testing...".format(ea_itera))
            X_train = np.concatenate((X_train, X_val), axis = 0)
            y_train = np.concatenate((y_train, y_val), axis = 0)
            y_train_data = np.concatenate((y_train_data, y_val_data), axis = 0)
            

            self.logger.info("Some statistics")
            print "Some statistics"
            statistics = {}
            for l in range(self.num_classes):
                statistics[l] = np.sum(y_train == l)
            
            self.logger.info("Statistics {}".format(statistics))
            print "Statistics {}".format(statistics)
        
            acc_net_val, acc_f1_val, percentage_test, deep_descriptor_test, y_used_test = self.test_dist(X_test, y_test, ea_itera, in_train = True)
        
            acc_dist_val = 0
            acc_dist_f1_val = 0
                
            
                
            
            del X_test
            del y_test
            del deep_descriptor_test
            del y_used_test
            
        
        del X_val
        del y_val
        del X_train
        del y_train
        
        
        return acc_net_val, acc_f1_val, acc_dist_val, acc_dist_f1_val, percentage_test
          
          
          
          
          
