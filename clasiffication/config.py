import os

class config:

    #some paths
    tensor_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Features_directory\pt_files'
    labelfile = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Image_list_new.csv'
    load_model_path = None


    #some parameters
    patch_size = 24
    epoch = 10
    learning_rate = 1e-4
    weight_decay = 1e-4
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    #model parameters
    first_dropout = 0.4
    middle_hidden_dimension = 1024
    output_hidden_dimension = 512
    last_dropout = 0.2
    num_labels = 2

    # dataloader parameters
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}     #原batch_size=16
    test_params =  {'batch_size': 8, 'shuffle': False, 'num_workers': 2}