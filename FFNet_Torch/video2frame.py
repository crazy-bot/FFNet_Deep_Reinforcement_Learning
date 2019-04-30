import cv2
import os

# Function to extract frames
def FrameCapture(folderPath,filename):
    # Path to video file
    filePath = os.path.join(folderPath, filename)
    print('reading -- ',filePath)
    vidObj = cv2.VideoCapture(filePath)
    storagePath = '/data/Tour20' +'/frames'+'/'+ filename

    if not os.path.exists(storagePath):
        os.makedirs(storagePath)
    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        # Saves the frames with frame-count
        cv2.imwrite(storagePath + "/frame%d.jpg" % count, image)

        count += 1
    print(storagePath)


# Driver Code
if __name__ == '__main__':
    # Calling the function
    folderPath = '/data/Tour20/Tour20-Videos/'
    # for filename in os.listdir(folderPath):
    #     FrameCapture(folderPath,filename)

    for root, subdirs, files in os.walk(folderPath):
        for subdir in subdirs:
            subFolderPath = os.path.join(folderPath, subdir)
            for filename in os.listdir(subFolderPath):
                FrameCapture(subFolderPath, filename)
