# -*- coding: utf-8 -*-
"""
Process a model to video folder 
"""
import logging
import time

import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import os
import pickle



def proc_vid_folder(folder, model_resolution='432x368', model='cmu', display=False):
    
    # Models 'cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small'
    
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda

    logger = logging.getLogger('TfPoseEstimator-Video')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh(model_resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    
    
    # Create target Directories if don't exist
    if not os.path.exists(folder + '/vid_out' ):
        os.mkdir(folder + '/vid_out')
    if not os.path.exists(folder + '/pkl_out' ):
        os.mkdir(folder + '/pkl_out')
    
    # Loop files over the folder
    for filename in os.listdir(folder):
        if filename.endswith(".mp4") and not os.path.isfile(folder + '/vid_out/' + filename + '_out.avi'): 

            logger.debug('Starting file : ' + filename)
            # captures the video file    
            cap = cv2.VideoCapture(folder + '/' + filename)
            
            # fps of video
            fps = cap.get(5)
            print('FPS = ', fps)
            fps_time = 0
            
            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            print('Resolution = ' + str(frame_width) + 'x' + str(frame_height) )
            
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(folder + '/vid_out/' + filename + '_out.avi',fourcc, fps, (frame_width,frame_height))
            
            human_list = []
            
            if cap.isOpened() is False:
                print("Error opening video stream or file")
                
            while cap.isOpened(): 
        
                ret_val, image = cap.read()
                
                if ret_val == True:
        
                    #humans = e.inference(image)
                    humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
                    #humans = e.inference(image, resize_to_default=(w > 0 and h > 0)) #, upsample_size=args.resize_out_ratio)
                        
                    # image + human
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                    
                    # Write the frame into the file 'output.avi'
                    out.write(image)
            
                    # Display the image
                    if display:
                        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imshow('tf-pose-estimation result', image)
                    
                    
                    fps_time = time.time()          
                   
                    tstamp = cap.get(0)
                    
                    # output the inference
                    human_list.append([tstamp /1000 , humans])
                    
                    if cv2.waitKey(1) == 27:
                        break
         
                # Break the loop
                else: 
                    break
        
        
            # save the poses in the pose dir
            pickle.dump(human_list, open(folder + '/pkl_out/' + filename +  '_poses.pkl', 'wb'))
                
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            
            
            logger.debug('finished file : ' + filename)


proc_vid_folder(folder='./sarahscottpole', model='cmu')