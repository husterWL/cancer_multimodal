import os

class config:

    #some paths
    tensor_path = '/mnt/Data/breast_cancer/pt_files'
    labelfile = '/mnt/Data/breast_cancer/Image_list_new.csv'
    emr_path = '/mnt/Data/breast_cancer/EMR.csv'
    img_path = '/mnt/Data/breast_cancer/patches'
    coords_path = '/mnt/Data/breast_cancer/h5_files'
    load_model_path = '/mnt/breast_cancer_multimodal/model/unimodal/pytorch_model_unimodal_univision_1111_1.bin'
    # load_model_path = '/mnt/breast_cancer_multimodal/model/multimodal/pytorch_model_multimodal_bicross_1111_1.bin'
    # load_model_path = '/mnt/breast_cancer_multimodal/model/checkpoint.pt'
    output_path = '/mnt/breast_cancer_multimodal/model'


    #some parameters
    patch_size = 24
    epoch = 10
    learning_rate = 5e-4
    scheduler_gamma = 0.95
    weight_decay = 1e-3
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    patience = 5

    #model parameters
    first_dropout = 0.4
    middle_hidden_dimension = 1024
    output_hidden_dimension = 512
    last_dropout = 0.3
    num_labels = 2

    #fusion parameters
    img_dimension = 1024
    emr_dimension = 106
    model_type = 'multimodel'
    fusion_type = 'Bicrossmodel'
    fusion_hidden_dimension = 512
    num_heads = 8
    fuse_dropout = 0.3
    attention_dropout = 0.2

    #attention visualization parameters
    


    # dataloader parameters
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
    test_params =  {'batch_size': 8, 'shuffle': False, 'num_workers': 2}
    heat_params =  {'batch_size': 4, 'shuffle': False, 'num_workers': 2}

    
    #EMR
    EMR_FEATURES = ['Patient ID', 
                 'Age', 'Gender', 'Disease Course Type', 'Personal Tumor History', 'Family Tumor History',
                 'Prophase Treatment', 'Neoadjuvant Chemotherapy', 'Dimple Sign', 'Orange Peel Appearance',
                 'Redness And Swelling Of Skin', 'Skin Ulcers', 'Tumor', 'Breast Deformation', 'Nipple Change',
                 'Nipple Discharge', 'Axillary Lymphadenectasis', 'Swelling Of Lymph Nodes', 'Tumor Position', 
                 'Tumor Number', 'Tumor Size', 'Tumor Texture', 'Tumor Border', 'Smooth Surface', 'Tumor Morphology',
                 'Activity', 'Capsules', 'Tenderness', 'Skin Adhesion', 'Pectoral Muscle Adhesion', 
                 'Diagnosis_Belnign_1_Malignant_2']