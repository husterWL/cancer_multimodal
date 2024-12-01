import os
import re
import csv

file_path = r'd:\BaiduNetdiskDownload\mutimodal_breast_cancer'

filenames = []

with open(r'D:\BaiduNetdiskDownload\mutimodal_breast_cancer\Structured data in EMR.csv') as csvfile:
    read = csv.reader(csvfile)
    for row in read:
        filenames.append(row[0].strip())
filenames = filenames[1:]
print(filenames)

for root, dirs, files in os.walk(file_path):
    for dir in dirs:
        # print(os.path.join(root, dir))
        for root2, dirs2, files2 in os.walk(os.path.join(root, dir)):
            for file in files2:
                # print(os.path.join(root2, file))
                id = re.search(r'_.*?_', file).group(0)[1:-1]
                if id not in filenames:
                    print(id)
                    print(os.path.join(root2, file))


                    filenames.append(id)

print(len(filenames))