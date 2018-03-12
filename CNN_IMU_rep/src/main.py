'''
Created on Mar 2, 2018

@author: fmoya
'''

'''
Created on Nov 24, 2017

@author: fmoya
'''


import logging

from Network_selecter import Network_selecter 


def configuration(dataset_idx, network_idx, output_idx, usage_modus_idx = 0, dataset_fine_tuning_idx = 0):
    
    dataset = {0 : 'locomotion', 1 : 'gesture', 2 : 'pamap2'}
    network = {0 : 'cnn', 1 : 'cnn_imu'}
    output = {0 : 'softmax'}
    usage_modus = {0 : 'train', 1 : 'test'}
    
    plotting = False
    fine_tunning = False
    model_fine_tuning = dataset[dataset_fine_tuning_idx]
    epochs_plus = 0
    GPU = 2
    
    
    if usage_modus[usage_modus_idx] == 'train'  or usage_modus[usage_modus_idx] == 'test': 
        NB_sensor_channels = {'locomotion' : 113, 'gesture' : 113, 'pamap2' : 40}
        sliding_window_length = {'locomotion' : 24, 'gesture' : 24, 'pamap2' : 100}
        sliding_window_step = {'locomotion' : 12, 'gesture' : 2, 'pamap2' : 22}
        num_classes = {'locomotion' : 5, 'gesture' : 18, 'pamap2' : 12}
        lr = {'locomotion' : {'cnn' : 0.0001, 'cnn_imu': 0.0001},
              'gesture' : {'cnn' : 0.001, 'cnn_imu': 0.0001},
              'pamap2' : {'cnn' : 0.001, 'cnn_imu': 0.0001}}
        epochs = {'locomotion' : {'cnn' : {'softmax' : 12},
                                  'cnn_imu' : {'softmax' : 12}},
                  'gesture' : {'cnn' : {'softmax' : 6},
                               'cnn_imu' : {'softmax' : 12}},
                  'pamap2' : {'cnn' : {'softmax' : 12},
                              'cnn_imu' : {'softmax' : 12}}}
        use_maxout = {'cnn' : False, 'cnn_imu': False}
        balancing =  {'locomotion' : False, 'gesture' : False, 'pamap2': False}
        filter_size =  {'locomotion' : 5, 'gesture' : 5, 'pamap2': 5}
        division_epochs =  {'locomotion' : 1, 'gesture' : 1, 'pamap2': 1}
        batch_size = {'cnn' : {'locomotion' : 100, 'gesture' : 100, 'pamap2' : 50},
                      'cnn_imu' : {'locomotion' : 100, 'gesture' : 100, 'pamap2' : 50}}
        
    
        num_filters = {'locomotion' : {'cnn' : 64, 'cnn_imu': 64},
                       'gesture' : {'cnn' : 64, 'cnn_imu': 64},
                       'pamap2' : {'cnn' : 64, 'cnn_imu': 64}}
        
        train_show = {'cnn' : 20, 'cnn_imu' :30}
        valid_show = {'cnn' : 100, 'cnn_imu' :250}


        
        if fine_tunning:
            folder_exp = 'fine_tuning'
            lr_mult = 1.0
    
            epochs = {'locomotion' : {'cnn' : {'softmax' : 6},
                                      'cnn_imu' : {'softmax' : 6}},
                      'gesture' : {'cnn' : {'softmax' : 10},
                                   'cnn_imu' : {'softmax' : 8}},
                      'pamap2' : {'cnn' : {'softmax' : 6},
                                  'cnn_imu' : {'softmax' : 6}}}
        else:
            if usage_modus[usage_modus_idx] == 'train_final':
                
                folder_exp = 'experiment_final'
                lr_mult = 1
            if usage_modus[usage_modus_idx] == 'train_random':
                
                folder_exp = 'experiment_random'
                lr_mult = 1
            else:
                folder_exp = 'experiment'
                lr_mult = 1
                        
        
    
    
    configuration = {'dataset' :dataset[dataset_idx],
                     'network' : network[network_idx],
                     'output' : output[output_idx],
                     'num_filters' : num_filters[dataset[dataset_idx]][network[network_idx]],
                     'filter_size' : filter_size[dataset[dataset_idx]],
                     'lr' : lr[dataset[dataset_idx]][network[network_idx]] * lr_mult,
                     'epochs' : epochs[dataset[dataset_idx]][network[network_idx]][output[output_idx]] + epochs_plus,
                     'train_show' : train_show[network[network_idx]],
                     'valid_show' : valid_show[network[network_idx]],
                     'plotting' : plotting,
                     'usage_modus' : usage_modus[usage_modus_idx],
                     'folder_exp' : folder_exp,
                     'fine_tunning' : fine_tunning,
                     'model_fine_tuning' : model_fine_tuning,
                     'use_maxout' : use_maxout[network[network_idx]],
                     'balancing' : balancing[dataset[dataset_idx]],
                     'GPU': GPU,
                     'division_epochs' : division_epochs[dataset[dataset_idx]],
                     'NB_sensor_channels' : NB_sensor_channels[dataset[dataset_idx]],
                     'sliding_window_length' : sliding_window_length[dataset[dataset_idx]],
                     'sliding_window_step' : sliding_window_step[dataset[dataset_idx]],
                     'batch_size' : batch_size[network[network_idx]][dataset[dataset_idx]],
                     'num_classes' : num_classes[dataset[dataset_idx]]}
    
    
    return configuration



def setup_experiment_logger(experiment_class, logging_level=logging.DEBUG, filename=None):
    # set up the logging
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
    if filename != None:
        logging.basicConfig(filename=filename,level=logging.DEBUG,
                            format=logging_format)
    else:
        logging.basicConfig(level=logging_level,
                            format=logging_format)
    logger = logging.getLogger(experiment_class.__class__.__name__)
    return logger



def main():
    
    config = configuration(dataset_idx = 2, network_idx = 0, output_idx = 0, usage_modus_idx = 1, dataset_fine_tuning_idx = 1)

    net_selecter = Network_selecter(config)
    
    logger = setup_experiment_logger(experiment_class = net_selecter, 
                                     filename='../' + config['dataset'] + '/' + config['network'] + '/' + config['output'] + '/' + config['folder_exp'] + '/logger_exp.txt')
    
    logger.info("Main: Starting training of Opportunity task with CNN_EA")
    
    
    net_selecter.set_logger(logger)
    
    net_selecter.net_selecter()
    
    
    
    
    
    return

if __name__ == '__main__':
    
    
    main()
    
    print "Done"
