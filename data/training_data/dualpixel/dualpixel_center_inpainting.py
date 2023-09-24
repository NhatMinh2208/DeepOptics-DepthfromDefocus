import numpy as np
import cv2 as cv
import os

def inpaint(directory, id):
    img = cv.imread(os.path.join(directory, "merged_depth",id, "result_merged_depth_center.png"),0)
    
    mask= cv.imread(os.path.join(directory, "merged_conf",id, "result_merged_conf_center.exr"),cv.IMREAD_UNCHANGED)
    #If the pixel value is larger than the threshold, it is set to 0, otherwise it is set to a maximum value
    ret,threshold = cv.threshold(img,0.01,255,cv.THRESH_BINARY_INV)
    #the threshhold masking which pixel need impainted, 3 is the imapainting radius
    dst = cv.inpaint(img, threshold, 3,cv.INPAINT_TELEA)

    try:
        os.makedirs(os.path.join(directory,"inpainted_depth", id ))
    except:
        print("Folder exists or can not be created ", os.path.join(directory,"inpainted_depth" ,id ))

    cv.imwrite(os.path.join(directory,"inpainted_depth", id, "result_merged_inpainted_depth_center.png" ), dst)
    cv.imwrite(os.path.join(directory,"inpainted_depth", id, "mask_inpainted.png" ), threshold)
    cv.imwrite(os.path.join(directory,"inpainted_depth", id, "img.png" ), img)



target = ["train","test"]
datadir = "./"

for dir in target:
    directory = os.path.join(datadir, dir)
    depth_dir = os.path.join(directory, "merged_depth")
    captlist =      [
        name for name in os.listdir(depth_dir)
        if os.path.isdir(os.path.join(depth_dir, name))
    ]    


    for id in captlist:
        print(id)
        inpaint(directory, id )
