import os 

path_list = []
train_path = 'data/coco128/images/train2017'
for file in os.listdir(train_path):
    if file.endswith('.jpg'):
        path_list.append(os.path.join(train_path, file))

with open('data/coco12/train.txt', 'w') as f:
    for path in path_list:
        f.write(path+ '\n')
