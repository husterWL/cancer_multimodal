import os
import re
import csv
import pandas as pd


file_path = r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Result_directory\stitches'

filenames = []
filenames2 = [] 

with open(r'D:\BaiduNetdiskDownload\multimodal_breast_cancer\EMR.csv') as csvfile:
    read = csv.reader(csvfile)
    for row in read:
        filenames.append(row[0].strip())
filenames = filenames[1:]
print(len(filenames))

for root, dirs, files in os.walk(file_path):
    for file in files:
        # print(os.path.join(root, dir))
        # for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
            # for file in files2:
                # print(os.path.join(root2, file))
        id = re.search(r'_.*?_', file).group(0)[1:-1]
        
        if id not in filenames:
            print(id)
            print(os.path.join(root, file))
        
        else:
            if id not in filenames2:
                filenames2.append(id)

print(len(filenames2))

for id in filenames:
    if id not in filenames2:
        print(id)

# frame = pd.DataFrame(columns=['case_id', 'slide_id', 'label'])

# for root, dirs, files in os.walk(file_path):
#     for file in files:
#         case_id = re.search(r'_.*?_', file).group(0)[1:-1]

#         if case_id in filenames2:
            
#             slide_id = re.match(r'^[^\.]+', file).group(0)
#             label = re.match(r'^[^\_]+', file).group(0)
#             print(case_id, slide_id, label)

#             # frame = frame.append({'case_id': case_id, 'slide_id': slide_id, 'label': label}, ignore_index=True)
#             frame_temp = pd.DataFrame({'case_id': case_id, 'slide_id': slide_id, 'label': label}, index=[0])
#             frame = pd.concat([frame, frame_temp])

# print(frame)

# frame.to_csv(r'D:\BaiduNetdiskDownload\mutimodal_breast_cancer\Image_list_new.csv')           

def process_csv(file_name):
    root = 'D:\BaiduNetdiskDownload\multimodal_breast_cancer\Result_directory'
    path = root + '\\' + file_name + '.csv'
    with open(path) as csvfile:
        read = csv.reader(csvfile)
        for row in read:
            if row[0].strip() == 'slide_id':
                continue
            slide_id = (re.match(r'^[^\.]+', row[0].strip()).group(0))
            # print(slide_id)
            jpg = root + '\\' +'stitches' + '\\' + slide_id + '.jpg'
            if not os.path.exists(jpg):
                print(slide_id)


process_csv('Malignant_pathological_image7')