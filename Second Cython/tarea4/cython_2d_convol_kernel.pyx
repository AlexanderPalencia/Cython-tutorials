import numpy
import cv2

cpdef unsigned char[:, :] convol2d_kernel_fast(unsigned char [:, :] image, double [:, :] kernel):

    # set the variable extension types
    cdef int x, y, w, h, kh, kw
    cdef unsigned char[:, :] imgPadding, subImg
    cdef double kernelResult
    
    # Creating empty image of the same size as image input
    resultImg = numpy.zeros_like(image)

    imgPadding = numpy.pad(image, 1, mode='constant')
    
    # grab the image and kernel dimensions
    h = image.shape[0]
    w = image.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    print('Cython cc size of original image y,x', h, w)
    print('Cython size of padding image y,x', imgPadding.shape[0], imgPadding.shape[1])
    print('Cython size of kernel', kh, kw)

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            subImg = imgPadding[y: y+kh, x: x+kw]
            kernelResult = numpy.multiply(subImg, kernel).sum()
            
            resultImg[y,x] = kernelResult

    
    cdef unsigned char[:, :] finalImg
            
    finalImg = cv2.normalize(resultImg, finalImg, 0,255, cv2.NORM_MINMAX) 
    return finalImg