import os 

def yaml_file(filename, save_path):
    path_list = []
    for file in os.listdir(filename):
        if file.endswith('.jpg'):
            path_list.append(os.path.join(filename, file))

    with open(save_path, 'w') as f:
        for path in path_list:
            f.write(path+ '\n')

yaml_file('drone_data/val', 'drone_data/val.txt')