import os
import json
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image

def search_best_frame(path_result):

    with open(path_result + ".pkl", 'rb') as f:
        data = pickle.load(f)

        max_list=[]
        max_index=[-1]*len(data)
        for i in range(len(data)):
            #print(class_list[i])
            num=[]
            max_value = 0

            for j in range(len(data[0])):
                count = 0
                for box in data[i][j]:
                    if max_value < box[4]:
                        max_value = box[4]
                        max_index[i] = j
                        max_box_num =count
                    count+=1
                num.append(count)
            max_list.append(max_value)

    a = sorted(range(len(max_list)), key = lambda k: max_list[k])

    return data[a[-1]][max_index[a[-1]]][count][0:4], max_index[a[-1]]

def create_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    video_frames = []

    if cap.isOpened():

        digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

        n = 0

        while True:
            ret, frame = cap.read()
            if ret:
                video_frames.append(frame)
                n += 1
            else:
                break
    return video_frames

def gen_config(args, detect_result = None):

    if args.seq != '' or args.video_based:
        # generate config from a sequence name

        seq_home = args.home
        result_home = 'results'


        if args.video_based:
            video_name = args.seq
            video_path = seq_home + '/' + video_name
            img_list = create_frames(video_path)
            init_bbox, first_frame = search_best_frame(detect_result)
            init_bbox[2] = init_bbox[2] - init_bbox[0]
            init_bbox[3] = init_bbox[3] - init_bbox[1]
            gt=None

        else:
            first_frame = 0
            seq_name = args.seq
            img_dir = os.path.join(seq_home, seq_name, 'img')
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

            img_list = os.listdir(img_dir)
            img_list.sort()
            img_list = [os.path.join(img_dir, x) for x in img_list]
            with open(gt_path) as f:
                gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
            init_bbox = gt[0]

        result_dir = os.path.join(result_home, args.seq)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, 'figs')
        result_path = os.path.join(result_dir, 'result.json')



    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''


    return img_list, init_bbox, gt, savefig_dir, args.display, result_path, first_frame
