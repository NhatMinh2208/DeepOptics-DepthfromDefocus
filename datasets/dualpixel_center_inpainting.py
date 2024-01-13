import numpy as np
import cv2 as cv
import os
# img = cv.imread('messi_2.jpg')
# mask = cv.imread('mask2.png',0)
# dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
# cv.imshow('dst',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

def inpaint(directory, id):
    img = cv.imread(os.path.join(directory, "merged_depth",id, "result_merged_depth_center.png"),0)
    
    mask= cv.imread(os.path.join(directory, "merged_conf",id, "result_merged_conf_center.exr"),cv.IMREAD_UNCHANGED)
    ret,threshold = cv.threshold(img,0.01,255,cv.THRESH_BINARY_INV)
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

