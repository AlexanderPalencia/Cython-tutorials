import numpy
import cv2

cpdef unsigned char[:, :] threshold_fast(unsigned char [:, :] image):

    # set the variable extension types
    cdef int x, y, w, h

    kernel = numpy.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    resultImg = numpy.zeros_like(image)
    
    imgPadding = numpy.pad(image, 1, mode='constant')
    
    # grab the image and kernel dimensions
    h = image.shape[0]
    w = image.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    print('size of original image y,x', h, w)
    print('size of padding image y,x', imgPadding.shape[0], imgPadding.shape[1])
    print('size of kernel', kh, kw)

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            subImg = imgPadding[y: y+kh, x: x+kw]
            resultImg[y,x] = numpy.multiply(subImg, kernel).sum()
            
    resultImg = cv2.normalize(resultImg, resultImg, 0,255, cv2.NORM_MINMAX) 
    return resultImg