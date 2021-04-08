import torchvision
import cv2
import torch
import argparse
import time
import segmentation_utils
from PIL import Image



# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
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
