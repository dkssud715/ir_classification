import cv2
import os
import numpy as np

def video2image():
    numbers = [2564, 2559, 2560, 2561, 2562, 2563, 1242, 1237, 1238, 1239, 1240, 1241, 2566, 2567, 2568, 2569, 2570, 2571, 2578, 2579, 2580, 2581, 2582, 2583, 1467, 1468, 1469, 1470, 1471, 1472, 1473]
    dir_path = os.getcwd()
    #'/content/drive/MyDrive/GasVid'
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    for n in numbers:
        #print(dir_path)
        video_name = dir_path + '/Videos/MOV_%d.mp4'%n
        vidcap = cv2.VideoCapture(video_name)
        print(video_name)
        success, back = vidcap.read()
        back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        #back = cv2.GaussianBlur(back, (0, 0), 1.0)
        #fback = back.astype(np.float32)
        #imgmask = fgbg.apply(image)
        count = 1
        new_path = dir_path + '/Images/%d' %n
        success, image = vidcap.read()
        while success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
            #imgmask = fgbg.apply(image)
            #print(image.shape)
            #print(fback.shape)
            #cv2.accumulateWeighted(gray, fback, 0.001)
            #back = fback.astype(np.uint8)
            #diff = cv2.absdiff(gray, back)
            #_, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            # cv2.imwrite(new_path + "/%05d.jpg" % count, diff)     # save frame as JPEG file
            cv2.imwrite(new_path + "/%05d.jpg" % count, gray)     # save frame as JPEG file
            count += 1
            success,image = vidcap.read()
        vidcap.release()
        #print('Read a new frame: ', success)
    """
    train = [2564, 2559, 2560, 2561, 2562, 2566, 2567, 2568, 2569, 2570, 2578, 2578, 2579, 2580, 2581, 2582]
    test = [1242, 1237, 1238, 1239, 1240, 1467, 1468, 1469, 1470, 1471]
    
    for n in train:
        dir_path = os.getcwd()
        #print(dir_path)
        video_name = dir_path + '/Videos/MOV_' + str(n) + '.mp4'
        vidcap = cv2.VideoCapture(video_name)
        success, image = vidcap.read()
        count = 1
        new_path = dir_path + '/train/%d' %n
        while success:
            cv2.imwrite(new_path + "/%05d.jpg" % count, image)     # save frame as JPEG file
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1
        #print('Read a new frame: ', success)

    for n in test:
        dir_path = os.getcwd()
        #print(dir_path)
        video_name = dir_path + '/Videos/MOV_' + str(n) + '.mp4'
        vidcap = cv2.VideoCapture(video_name)
        success, image = vidcap.read()
        count = 1
        new_path = dir_path + '/test/%d' %n
        while success:
            cv2.imwrite(new_path + "/%05d.jpg" % count, image)     # save frame as JPEG file
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1
    #print("finish! convert video to frame {name}".format(name=video_name))
    """
    print("all convert finish!!") 

def makedirectory():
    train = [2564, 2559, 2560, 2561, 2562, 2566, 2567, 2568, 2569, 2570, 2578, 2578, 2579, 2580, 2581, 2582]
    test = [1242, 1237, 1238, 1239, 1240, 1467, 1468, 1469, 1470, 1471]
    numbers = [2564, 2559, 2560, 2561, 2562, 2563, 1242, 1237, 1238, 1239, 1240, 1241, 2566, 2567, 2568, 2569, 2570, 2571, 2578, 2579, 2580, 2581, 2582, 2583, 1467, 1468, 1469, 1470, 1471, 1472, 1473]

    dir_path = os.getcwd()
    
    for n in numbers:
        new_path = dir_path + '/Images/%d'%n
        os.mkdir(new_path)


    """
    for n in train:
        new_path = dir_path + '/train/' + str(n)
        os.mkdir(new_path)
    for n in test:
        new_path = dir_path + '/test/' + str(n)
        os.mkdir(new_path)
    """

if __name__ == '__main__':
    makedirectory()
    video2image()