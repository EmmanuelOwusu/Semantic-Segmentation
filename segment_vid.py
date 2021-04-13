import torchvision
import cv2
import torch
import argparse
import time
import segmentation_utils
from PIL import Image


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='inputs')
args = vars(parser.parse_args())


# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download or load the model from disk
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
# load the model onto the computation device
model = model.eval().to(device)

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')



# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.avi", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second



####Applying Deep Learning Image Segmentation to Each Video Frame

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            outputs = segmentation_utils.get_segment_labels(frame, model, device)
        
        # draw boxes and show current frame on screen
        segmented_image = segmentation_utils.draw_segmentation_map(outputs['out'])
        final_image = segmentation_utils.image_overlay(frame, segmented_image)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # press `q` to exit
        wait_time = max(1, int(fps/4))
        cv2.imshow('image', final_image)
        out.write(final_image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
out.release()

# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")