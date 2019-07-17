import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import Human, TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common
from tf_pose.common import CocoPart

import os
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='656x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--flip', type=bool, default=False, help='True to flip the video horizontal')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    
    
    # fps of video
    fps = cap.get(5)
    print(fps)
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.video + '_out.avi',fourcc, fps, (frame_width,frame_height))
    
    human_list = []
    
#    prev_humans = [ Human(pairs = []) for i in range(100) ]

    if cap.isOpened() is False:
        print("Error opening video stream or file")
        
    while cap.isOpened(): 

        ret_val, image = cap.read()
        
        if ret_val == True:

            #humans = e.inference(image)
            if args.flip:
                image = cv2.flip(image, 0)
            humans = e.inference(image, resize_to_default=True, upsample_size=8.0)
            #humans = e.inference(image, resize_to_default=(w > 0 and h > 0)) #, upsample_size=args.resize_out_ratio)

            
            # human motiv, retain last part
            
#            for h, human in enumerate(humans):
#            # human 1 only
#                if prev_humans[h] is not None:
#                    prev_h = prev_humans[h]
#                    for i in range(common.CocoPart.Background.value):
#                        if i not in human.body_parts.keys():
#                            if i in prev_h.body_parts.keys():
#                                human.body_parts[i] = prev_h.body_parts[i]
#                            
#            prev_humans = humans
            
            # image + human
            if not args.showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            
            # Write the frame into the file 'output.avi'
            out.write(image)
    
            # Display the image
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            
            
            fps_time = time.time()          
#            videotime = frame / fps
#            print(videotime)
            
            tstamp = cap.get(0)
            
            # output the inference
            human_list.append([tstamp /1000 , humans])
            
            if cv2.waitKey(1) == 27:
                break
 
        # Break the loop
        else: 
            break


    # save the poses
    pickle.dump(human_list, open( args.video + '_poses.pkl', 'wb'))
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


    
logger.debug('finished+')
