import numpy
import cv2

def convol2d_kernel_slow(image, kernel):
    print('Python apply kernel function.')
    # Creating empty image of the same size as image input
    resultImg = numpy.zeros_like(image)

    # Adding padding to image to pass kernel and return and image of the same size
    imgPadding = numpy.pad(image, 1, mode='constant')
    
    # grab the image and kernel dimensions
    h = image.shape[0]
    w = image.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # print('size of original image y,x', h, w)
    # print('size of padding image y,x', imgPadding.shape[0], imgPadding.shape[1])
    # print('size of kernel', kh, kw)

    # loop over the image and apply kernel to every subimage
    for y in range(0, h):
        for x in range(0, w):
            # Slicing in image to get subimage
            subImg = imgPadding[y: y+kh, x: x+kw]
            # apply kernel operation to sub image
            resultImg[y,x] = numpy.multiply(subImg, kernel).sum()
    
    # Normlalize output to values between 0 and 255
    resultImg = cv2.normalize(resultImg, resultImg, 0,255, cv2.NORM_MINMAX) 
    return resultImg