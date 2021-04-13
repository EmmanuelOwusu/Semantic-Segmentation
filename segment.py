import torchvision
import numpy
import torch
import argparse
import segmentation_utils
import cv2
from PIL import Image



# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input')
args = vars(parser.parse_args())



# download or load the model from disk
# $python -c 'import torchvision;torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)'
# Downloading: "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth" to /home/moto/.cache/torch/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth
# 100.0%
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model to eval() model and load onto computation devicce
model.eval().to(device)

# read the image
image = Image.open(args['input'])
# do forward pass and get the output dictionary
outputs = segmentation_utils.get_segment_labels(image, model, device)
# get the data from the `out` key
outputs = outputs['out']
segmented_image = segmentation_utils.draw_segmentation_map(outputs)


##Now we can overlay the segmented mask on top the original image.

final_image = segmentation_utils.image_overlay(image, segmented_image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# show the segmented image and save to disk
cv2.imshow('Segmented image', final_image)
cv2.waitKey(0)
cv2.imwrite(f"outputs/{save_name}.jpg", final_image)