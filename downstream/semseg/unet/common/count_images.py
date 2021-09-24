import os


filelist = '/checkpoint/jihou/data/scannet/rgbd/splits/scannetv2_train.txt'
image_folder = '/checkpoint/jihou/data/scannet/rgbd/'
scenes = open(filelist, 'r').readlines()
percentage = [0.01, 0.05, 0.1]
total_num_images = 19466

current_num = 0
percentage_idx = 0
writelines = []
for scene in scenes:
    images = os.listdir(os.path.join(image_folder, scene.strip(), 'color'))
    current_num += len(images)
    if current_num >= percentage[percentage_idx] * total_num_images:
        f = open('scannetv2_train_{}.txt'.format(int(percentage[percentage_idx]*100)), 'w')
        f.writelines(writelines)
        percentage_idx += 1
    writelines.append(scene)

