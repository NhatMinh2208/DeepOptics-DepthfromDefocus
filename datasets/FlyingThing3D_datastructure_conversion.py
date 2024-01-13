import glob
import os
import shutil

#1 converting the disparity



total_files_dis =0
for s in ['TRAIN', 'TEST']:
    target = os.path.join('./', 'FlyingThings3D_subset', s.lower(), 'disparity', 'right')
    print("Target directory: ", target)
    try:
        os.makedirs(target)
    except:
        print("Folder exists or can not be created ", target)

    for abc in ['A','B','C']:
        for id in range(0, 750):
            idstr  = '{:04d}'.format(id)
            source = os.path.join('./Original/disparity',s,abc,idstr,'right')
            fileprefix = abc+"_"+idstr

            print("check folder: ", source)
            listfiles = glob.glob(source+"/*.pfm")
            total_files_dis+= len(listfiles)

            for f in listfiles:
                basefile = os.path.basename(f)
                newfilename = fileprefix+ basefile
                print("copy from ", f , " to new target file ", newfilename)
                shutil.copy(f, target+'/'+ newfilename)

#2 converting the image
total_files_img =0
for s in ['TRAIN', 'TEST']:
    target = os.path.join('./', 'FlyingThings3D_subset', s.lower(), 'image_clean', 'right')
    print("Target directory: ", target)
    try:
        os.makedirs(target)
    except:
        print("Folder exists or can not be created ", target)

    for abc in ['A','B','C']:
        for id in range(0, 750):
            idstr  = '{:04d}'.format(id)
            source = os.path.join('./Original/frames_cleanpass',s,abc,idstr,'right')
            fileprefix = abc+"_"+idstr

            print("check folder: ", source)
            listfiles = glob.glob(source+"/*.png")
            total_files_img+= len(listfiles)

            for f in listfiles:
                basefile = os.path.basename(f)
                newfilename = fileprefix+ basefile
                print("copy from ", f , " to new target file ", newfilename)
                shutil.copy(f, target+'/'+ newfilename)






print("A total of image files ", total_files_img, " copied to the new place!")
print("A total of disparity files ", total_files_dis, " copied to the new place!")