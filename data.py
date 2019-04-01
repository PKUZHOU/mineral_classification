import torch
import cv2
import os
import random
import torch.utils.data as data
import numpy as np


def rotate(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def random_crop(image):
    # random crop the image, while keeping the center fixed.
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    min_wh = min(w, h)
    width = random.uniform(0.5 * min_wh, min_wh)
    xmin = int(center[0] - width / 2)
    ymin = int(center[1] - width / 2)
    xmax = int(center[0] + width / 2)
    ymax = int(center[1] + width / 2)
    return image[ymin:ymax, xmin:xmax, :]


def flips(image):
    # horizontal, vertical, and both
    h_flip = cv2.flip(image, 1)
    v_flip = cv2.flip(image, 0)
    hv_flip = cv2.flip(image, -1)
    return [image, h_flip, v_flip, hv_flip]


def gen_samples(image_path, num):
    cv_image = cv2.imread(image_path)
    samples = []
    for i in range(num):
        croped = random_crop(cv_image)
        resized = cv2.resize(croped, (224, 224))
        flipeds = flips(resized)
        samples += flipeds
    return samples


def create_dataset(root):
    ori_root = os.path.join(root,'data')
    train_dir = os.path.join(root,'train')
    val_dir = os.path.join(root,'val')

    for p in [train_dir,val_dir]:
        if(not os.path.isdir(p)):
            os.mkdir(p)

    minerals_cnt = {}
    image_paths = {}
    folds = os.listdir(ori_root)
    for fold in folds:
        fold_path = os.path.join(ori_root, fold)
        minerals = os.listdir(fold_path)
        for mineral in minerals:
            # for generating training images' ids
            if (mineral not in minerals_cnt.keys()):
                minerals_cnt[mineral] = 0
            if (mineral not in image_paths.keys()):
                image_paths[mineral] = []

            mineral_path = os.path.join(fold_path, mineral)
            scales = os.listdir(mineral_path)
            for scale in scales:
                scale_path = os.path.join(mineral_path, scale)
                images = os.listdir(scale_path)
                for image in images:
                    image_path = os.path.join(scale_path, image)
                    image_paths[mineral].append(image_path)
    print minerals_cnt
    # gen_testset
    test_number = 20
    cnt = 0
    val_list = open("val.txt", 'w')
    for k in image_paths.keys():
        print k, len(image_paths[k])
        random.shuffle(image_paths[k])
        for i in range(test_number):
            images = gen_samples(image_paths[k][i],10)
            for image in images:
                save_path = os.path.join(val_dir,str(cnt)+'.jpg')
                cnt+=1
                cv2.imwrite(save_path,image)
                val_list.write(save_path+' '+str(image_paths.keys().index(k))+'\n')
                print save_path
    val_list.close()

    train_list = open("train.txt", 'w')
    train_number = 80
    #gen_trainset
    cnt = 0
    for k in image_paths.keys():
        for i in range(train_number):
            true_index = i%(len(image_paths[k])-test_number)+test_number
            images = gen_samples(image_paths[k][true_index],10)
            for image in images:
                save_path = os.path.join(train_dir,str(cnt)+'.jpg')
                cnt+=1
                cv2.imwrite(save_path,image)
                train_list.write(save_path+' '+str(image_paths.keys().index(k))+'\n')
                print save_path

    train_list.close()


class MyDataset(data.Dataset):
    def __init__(self, lists, train=True):
        self.lists_file = lists
        self.labels = {}
        self.lists = self.get_list()

    def get_list(self):
        with open(self.lists_file, 'r') as f:
            lists = f.readlines()
        return lists

    def __getitem__(self, idx):
        anno = self.lists[idx].strip('\n')
        path, label = anno.split(' ')
        image = cv2.imread(path).astype(np.float32)
        image = torch.from_numpy(image)
        image = (image - 127.5) / 127.5
        image = image.permute(2, 0, 1)
        label = torch.Tensor([int(label)]).long()
        return image, label

    def __len__(self):
        return len(self.lists)


if __name__ == '__main__':
    root = "/home/zhou/mineral_data"
    create_dataset(root)