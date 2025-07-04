import os

class config:

    #some paths
    tensor_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Features_directory\pt_files'
    labelfile = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Image_list_new.csv'
    emr_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\EMR.csv'
    # load_model_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\_model_new\unimodal\pytorch_model_multimodal_bicrossmodel_0702_1.bin'
    load_model_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\_model_new\multimodal\pytorch_model_multimodal_bicrossmodel_0702_1.bin'
    # output_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\_model'
    output_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\_model_new'
    all_data_path = r'./data/data.json'


    #some parameters
    patch_size = 24
    epoch = 10
    learning_rate = 1e-4
    weight_decay = 1e-4
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    patience = 5

    #model parameters
    first_dropout = 0.4
    middle_hidden_dimension = 1024
    output_hidden_dimension = 512
    last_dropout = 0.2
    num_labels = 2

    #fusion parameters
    img_dimension = 1024
    emr_dimension = 106
    model_type = 'unimodal'
    fusion_type = 'Bicrossmodel'
    fusion_hidden_dimension = 512
    num_heads = 8
    fuse_dropout = 0.3
    attention_dropout = 0.3

    # dataloader parameters
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
    test_params =  {'batch_size': 2, 'shuffle': False, 'num_workers': 2}
    
    #EMR
    EMR_FEATURES = ['Patient ID', 
                 'Age', 'Gender', 'Disease Course Type', 'Personal Tumor History', 'Family Tumor History',
                 'Prophase Treatment', 'Neoadjuvant Chemotherapy', 'Dimple Sign', 'Orange Peel Appearance',
                 'Redness And Swelling Of Skin', 'Skin Ulcers', 'Tumor', 'Breast Deformation', 'Nipple Change',
                 'Nipple Discharge', 'Axillary Lymphadenectasis', 'Swelling Of Lymph Nodes', 'Tumor Position', 
                 'Tumor Number', 'Tumor Size', 'Tumor Texture', 'Tumor Border', 'Smooth Surface', 'Tumor Morphology',
                 'Activity', 'Capsules', 'Tenderness', 'Skin Adhesion', 'Pectoral Muscle Adhesion', 
                 'Diagnosis_Belnign_1_Malignant_2']