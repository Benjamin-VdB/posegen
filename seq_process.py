# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:22:09 2019

@author: ben.vdb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:04:06 2019

@author: ben.vdb
"""

import pickle
import pandas as pd
import numpy as np
import os

import cv2

from tf_pose import common
from tf_pose.common import CocoPart, CocoPairs


def pickle_to_sequence(file, lookback, steps, forward, lag, retain_previous=True, standardize=False, normalize=True, head=False, add_flip=False, seq2seq=False):
    
    if head:
        point_list=range(18)

    else:
        point_list=range(1,14)
        
    nb_points=len(point_list)
    
    # load pickle
    humans = pd.read_pickle(file)
    
    if add_flip and os.path.isfile(file.split('pkl_out/')[0] + 'pkl_flip_out/' + file.split('pkl_out/')[1] ):
        humans_flip =  pd.read_pickle(file.split('pkl_out/')[0] + 'pkl_flip_out/' + file.split('pkl_out/')[1])
    else:
        add_flip = False
    
    # numpy matrix init      
    # seq_data = np.zeros((len(humans), 18,2))
    seq_data = np.zeros((len(humans), nb_points*2))
    # seq_data = np.empty((len(humans), nb_points*2)) * np.nan

    
    # loop over timesteps
    for t, human in enumerate(humans):
    
        # human 1 only if it exists, otherwise the previous time step
        if human[1] != []:
            # body part if it exists, otherwise the previous coords
            for i, p in enumerate(point_list):
                if p in human[1][0].body_parts.keys():
                    # seq_data[t,i,0] = humans[t][1][0].body_parts[i].x
                    # seq_data[t,i,1] = humans[t][1][0].body_parts[i].y
                    seq_data[t,i] = humans[t][1][0].body_parts[p].x
                    seq_data[t,i+nb_points] = humans[t][1][0].body_parts[p].y
                                  
#                elif add_flip and humans_flip[t][1] != [] and p in humans_flip[t][1][0].body_parts.keys() and humans_flip[t][1][0].body_parts[p].score > 0.5:
#                    seq_data[t,i] = humans_flip[t][1][0].body_parts[p].x
#                    seq_data[t,i+nb_points] = 1 - humans_flip[t][1][0].body_parts[p].y
                    
                elif t !=0 and retain_previous:
                    # seq_data[t,i,0] = seq_data[t-1,i,0]
                    # seq_data[t,i,1] = seq_data[t-1,i,1]
                    seq_data[t,i] = seq_data[t-1,i]
                    seq_data[t,i+nb_points] = seq_data[t-1,i+nb_points]
#                else:
#                    # seq_data[t,i,0] = 0.
#                    # seq_data[t,i,1] = 0.
#                    seq_data[t,i] = 0.
#                    seq_data[t,i+nb_points] = 0.
                    
        elif add_flip and humans_flip[t][1] != []:
            for i, p in enumerate(point_list):
                if p in humans_flip[t][1][0].body_parts.keys():
                    seq_data[t,i] = humans_flip[t][1][0].body_parts[p].x
                    seq_data[t,i+nb_points] = 1 - humans_flip[t][1][0].body_parts[p].y
                
        else:
            for i, p in enumerate(point_list):
                if t != 0 and retain_previous:
                    # seq_data[t,i,0] = seq_data[t-1,i,0]
                    # seq_data[t,i,1] = seq_data[t-1,i,1]
                    seq_data[t,i] = seq_data[t-1,i]
                    seq_data[t,i+nb_points] = seq_data[t-1,i+nb_points]
    
                    
    # Standardization
    if standardize:
        mean_x = seq_data[:,:nb_points].mean()
        seq_data[:,:nb_points] -= mean_x
        mean_y = seq_data[:,nb_points:].mean()
        seq_data[:,nb_points:] -= mean_y
        
        std_x = seq_data[:,:nb_points].std()
        seq_data[:,:nb_points] /= std_x             
        std_y = seq_data[:,nb_points:].std()
        seq_data[:,nb_points:] /= std_y    

    # Normalisation
    if normalize:
        seq_data = normalize_seq(seq_data)
    
    
    # number of samples
    ref_points = np.arange(lookback, len(seq_data) - forward , lag)
    
    # init samples array, lookback // steps points before
    samples = np.zeros( (len(ref_points), lookback // steps , nb_points*2) )


    if seq2seq:
        
        targets = np.zeros( (len(ref_points), forward // steps , nb_points*2) )
    
        # loop over the samples
        for n, i in enumerate(ref_points):
            # indices for the lookback period, including the ref point
            indices = range(i - (lookback // steps - 1 )*steps, i + steps , steps)
            indices2 = range(i+1 , i + (forward // steps )*steps +1, steps)
            samples[n, : , : ] = seq_data[ indices, : ]
            targets[n, : , : ] = seq_data[ indices2, : ]

    else:
        targets = np.zeros( (len(ref_points), nb_points*2) )
        
        # loop over the samples
        for n, i in enumerate(ref_points):
            # indices for the lookback period, including the ref point
            indices = range(i - (lookback // steps - 1 )*steps, i + steps , steps)
            samples[n, : , : ] = seq_data[ indices, : ]
            targets[n, :  ] = seq_data[ i + forward, : ]
        
    
    return seq_data, samples, targets


#seq_data, samples, targets = pickle_to_sequence(file='./sarahscottpole/pkl_out/2018-08-16_14-32-57_UTC.mp4_poses.pkl', 
#                                               lookback=30, steps=1, forward=1, lag=10, retain_previous=False, 
#                                               standardize=False, normalize=True, head=False,
#                                               add_flip=True)
#draw_sequence(seq_data)




#seq_data, samples, targets = pickle_to_seq2seq(file='./sarahscottpole/pkl_out/2017-10-23_16-02-46_UTC.mp4_poses.pkl', 
#                                               lookback=300, steps=10, forward=60, lag=10,retain_previous=True, 
#                                               standardize=False, normalize=True, head=False)
#seq_data2, samples, targets = pickle_to_seq2seq(file='./sarahscottpole/pkl_out/2017-10-23_16-02-46_UTC.mp4_poses.pkl', 
#                                                   lookback=300, steps=10, forward=60, lag=10,retain_previous=True, 
#                                                   standardize=True, normalize=True, head=False)

def folders_to_sequence(folder_list, lookback, steps, forward, lag, retain_previous, standardize, normalize, head, add_flip, seq2seq):
    
    i = 0 

    for folder in folder_list:
        print(folder)
        # Loop files over the folder
        for filename in os.listdir(folder):
            if filename.find('_out.avi') >= 0:
    
                print(filename)
                pkl_filename = filename.split('_out.avi')[0] + '_poses.pkl'
                pkl_folder = folder.split('vid_out/')[0] + 'pkl_out/'
                
               
                seq_data, samples, targets = pickle_to_sequence(file=pkl_folder + pkl_filename,
                                                               lookback=lookback, steps=steps, forward=forward, lag=lag,
                                                               retain_previous=retain_previous, standardize=standardize,
                                                               normalize=normalize, head=head, add_flip=add_flip, seq2seq=seq2seq)               
                
           
                if i == 0:
                    merged_samples = samples
                    merged_targets = targets
                    merged_seq = seq_data
                    i += 1
                else:
                    merged_samples = np.append(merged_samples, samples, axis=0)
                    merged_targets = np.append(merged_targets, targets, axis=0)
                    merged_seq = np.append(merged_seq, seq_data, axis=0)
                
    return merged_seq, merged_samples, merged_targets



def normalize_seq(seq):
    
    nb_points = int(len(seq[0])/2)
    min_x = seq[:,:nb_points].min()
    min_y = seq[:,nb_points:].min()
    max_x = seq[:,:nb_points].max()
    max_y = seq[:,nb_points:].max()
    
    for j in range(nb_points):
        seq[:,j] = (seq[:,j] - min_x) / (max_x - min_x)
        seq[:,j+nb_points] = (seq[:,j+nb_points] - min_y) / (max_y - min_y)
            
    return seq

def normalize_vec(vec):
    
    nb_points = int(len(vec)/2)
    min_x = vec[:nb_points].min()
    min_y = vec[nb_points:].min()
    max_x = vec[:nb_points].max()
    max_y = vec[nb_points:].max()
    
    for j in range(nb_points):
        vec[j] = (vec[j] - min_x) / (max_x - min_x)
        vec[j+nb_points] = (vec[j+nb_points] - min_y) / (max_y - min_y)
            
    return vec


NoHeadPairs = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (0, 10),
    (10, 11), (11, 12), (1, 0) ]

def draw_squeleton(npimg, squel_vec, imgcopy=False):
                
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    
    nb_points = int(len(squel_vec)/2)
    
    if nb_points == 13:
        pairs = NoHeadPairs
    else:
        pairs = common.CocoPairsRender
    
    min_x = squel_vec[:nb_points].min()
    min_y = squel_vec[nb_points:].max()
    
    for i in range(nb_points):
        if squel_vec[i] > min_x + 0.01 and squel_vec[i+nb_points] > min_y + 0.01:
#        if i not in human.body_parts.keys():
#            continue

            #body_part = human.body_parts[i]
            center = (int(squel_vec[i] * image_w ), int(squel_vec[i+nb_points] * image_h ))
            #center = (int((squel_vec[i] + max_x)/max_x * 0.5 * image_w + 0.5), int((squel_vec[i+18] + max_y)/max_y * 0.5 * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=2, lineType=8, shift=0)

    # draw line
    for pair_order, pair in enumerate(pairs):
#        if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
#            continue

        # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
        if pair[0] in centers and pair[1] in centers:
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 2)

    return npimg

#npimg = np.zeros([480,240,3], dtype=np.uint8)
#
#npimg = draw_squeleton(npimg, samples[10,40,:], imgcopy=False)
#cv2.imshow('squeleton', npimg); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)


def draw_sequence(seq, normalize=True):
    
    if normalize:
        seq = normalize_seq(seq)
    
    black_img = np.zeros([640,480,3], dtype=np.uint8)
    for i in range(len(seq)):
        npimg = draw_squeleton(black_img, seq[i,:], imgcopy=True)
        cv2.imshow('squeleton', npimg) ; #cv2.waitKey(0); cv2.destroyAllWindows(); #cv2.waitKey(1)
        cv2.waitKey(33) # 30FPS
        if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()       
    
#draw_sequence(seq_data)
