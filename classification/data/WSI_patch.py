import h5py
import os



file_path = r'C:\Users\WL\Desktop\WSI\RESULT_DIRECTORY\patches'

filenames = []

for root, dirs, files in os.walk(file_path):
    for file in files:
        filenames.append(os.path.join(root, file))

for filename in filenames:
    print(filename)
    with h5py.File(filename, 'r') as f:
        # for key in f.keys():
        #     print(f[key], key, f[key].name)

        print("Keys: %s" % f.keys())    #Keys: <KeysViewHDF5 ['coords']>
        dataset = f['coords']           #<HDF5 dataset "coords": shape (45, 2), type "<i8"> 对应左上角的坐标
        data = dataset[:]
        # 打印数据
        print(data)
        