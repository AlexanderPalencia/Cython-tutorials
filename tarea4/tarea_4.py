# Manuel Alexander Palencia Gutierrez
import numpy as np
import cv2
import cython_2d_convol_kernel
import python_2d_convol_kernel
import time

"""
The SPEED UP CYTHON Vs python is 19 faster. This mean than or function in cython is 19 faster.
For the video we can see a great improvment using cython the speed up is gratter than 23.
"""

def sharp_effect_cython_from_path(imgPath):
    """Pass sharp Kernel Effect to cython 2d convolution
    Args:
        img (Numpy Array): The numpy array of a image.
    Return:  Sharpen img (Numpy Array memory): The numpy array of a image of the same size of image
    """
    # Read image BGR
    image = cv2.imread(imgPath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Creating Sharp kernel
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]], dtype=np.float32)
    # kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

    # Split RGB image into channels
    channel0Padding = image[:,:,0]
    channel1Padding = image[:,:,1]
    channel2Padding = image[:,:,2]

    # Creating List of Channels to iterate over
    channels = [channel0Padding, channel1Padding, channel2Padding]

    # Creating list to append kernel result of channels
    finalChannels = []

    # Iteration over channel list
    for channel in channels:
        kernelResult = cython_2d_convol_kernel.convol2d_kernel_fast(channel, kernel)
        finalChannels.append(np.asarray(kernelResult))

    # Merge the 3 channels to make a BGR image    
    mergedChannelsBGR = cv2.merge([finalChannels[0], finalChannels[1], finalChannels[2]])

    # Saving Imag
    # cv2.imwrite("./imgs/sharpImgCython.png", mergedChannelsBGR)
    return mergedChannelsBGR

def sharp_effect_python_from_path(imgPath):
    """Pass sharp Kernel Effect to python 2d convolution
    Args:
        img (Numpy Array): The numpy array of a image.
    Return:  Sharpen img (Numpy Array memory): The numpy array of a image of the same size of image
    """
    # Read image BGR
    image = cv2.imread(imgPath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Creating Sharp kernel
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])

    # Split RGB image into channels
    channel0Padding = image[:,:,0]
    channel1Padding = image[:,:,1]
    channel2Padding = image[:,:,2]

    # Creating List of Channels to iterate over
    channels = [channel0Padding, channel1Padding, channel2Padding]

    # Creating list to append kernel result of channels
    finalChannels = []

    # Iteration over channel list
    for channel in channels:
        finalChannels.append(python_2d_convol_kernel.convol2d_kernel_slow(channel, kernel))
    
    # Merge the 3 channels to make a BGR image    
    mergedChannelsBGR = cv2.merge([finalChannels[0], finalChannels[1], finalChannels[2]])

    # Saving Imag
    # a = cv2.imwrite("./imgs/sharpenImagePython.png", mergedChannelsBGR)
    return mergedChannelsBGR

def sharp_effect_cython_from_matrix(image):
    """Pass sharp Kernel Effect to cython 2d convolution
    Args:
        img (Numpy Array): The numpy array of a image.
    Return:  Sharpen img (Numpy Array memory): The numpy array of a image of the same size of image
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Creating Sharp kernel
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]], dtype=np.float32)
    # kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

    # Split RGB image into channels
    channel0Padding = image[:,:,0]
    channel1Padding = image[:,:,1]
    channel2Padding = image[:,:,2]

    # Creating List of Channels to iterate over
    channels = [channel0Padding, channel1Padding, channel2Padding]

    # Creating list to append kernel result of channels
    finalChannels = []

    # Iteration over channel list
    for channel in channels:
        kernelResult = cython_2d_convol_kernel.convol2d_kernel_fast(channel, kernel)
        finalChannels.append(np.asarray(kernelResult))

    # Merge the 3 channels to make a BGR image    
    mergedChannelsBGR = cv2.merge([finalChannels[0], finalChannels[1], finalChannels[2]])

    # Saving Imag
    # cv2.imwrite("./imgs/sharpImgCython.png", mergedChannelsBGR)
    return mergedChannelsBGR

def sharp_effect_python_from_matrix(image):
    """Pass sharp Kernel Effect to python 2d convolution
    Args:
        img (Numpy Array): The numpy array of a image.
    Return:  Sharpen img (Numpy Array memory): The numpy array of a image of the same size of image
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Creating Sharp kernel
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])

    # Split RGB image into channels
    channel0Padding = image[:,:,0]
    channel1Padding = image[:,:,1]
    channel2Padding = image[:,:,2]

    # Creating List of Channels to iterate over
    channels = [channel0Padding, channel1Padding, channel2Padding]

    # Creating list to append kernel result of channels
    finalChannels = []

    # Iteration over channel list
    for channel in channels:
        finalChannels.append(python_2d_convol_kernel.convol2d_kernel_slow(channel, kernel))
    
    # Merge the 3 channels to make a BGR image    
    mergedChannelsBGR = cv2.merge([finalChannels[0], finalChannels[1], finalChannels[2]])

    # Saving Imag
    return mergedChannelsBGR

def sharp_effect_video_python(videoPath):
    """Pass sharp Kernel Effect to video generate as output the same video but with the effect.
    Args:
        videoPath (Numpy Array): The numpy array of a image.
    Return:  Sharpen img (Numpy Array memory): The numpy array of a image of the same size of image
    """
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(videoPath)
    
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]], dtype=np.float32)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    listImages = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            # cv2.imshow('Frame',frame)

            # PASS FILTER
            listImages.append(sharp_effect_python_from_matrix(frame))

            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return listImages

def sharp_effect_video_cython(videoPath):
    """Pass sharp Kernel Effect to video generate as output the same video but with the effect.
    Args:
        videoPath (Numpy Array): The numpy array of a image.
    Return:  Sharpen img (Numpy Array memory): The numpy array of a image of the same size of image
    """
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(videoPath)
    
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]], dtype=np.float32)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    listImages = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            # cv2.imshow('Frame',frame)

            # PASS FILTER
            listImages.append(sharp_effect_cython_from_matrix(frame))

            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return listImages


if __name__ == '__main__':
    # "./imgs/test_train.jpg"

    start = time.time()
    a = sharp_effect_python_from_path("./imgs/test_train.jpg")
    end =  time.time()
    py_time = end - start
    print("Python time to run a image and apply kernel is {}".format(py_time))

    start = time.time()
    b = sharp_effect_cython_from_path("./imgs/test_train.jpg")
    end =  time.time()
    cy_time = end - start
    print("Cython time to run a image is {}".format(cy_time))

    print("Speedup = {}".format(py_time / cy_time))
    print('Results Python and Cython image are the same?? ', np.array_equal(a,b))




    start = time.time()
    c = sharp_effect_video_python('./videos/myVideo.avi')
    end =  time.time()
    py_time = end - start
    print("Python time to run a video and apply kernel is {}".format(py_time))



    start = time.time()
    d = sharp_effect_video_cython("./imgs/test_train.jpg")
    end =  time.time()
    cy_time = end - start
    print("Cython time to run a video and apply kernel is {}".format(cy_time))


    print("Speedup = {}".format(py_time / cy_time))