import numpy
import cv2

cpdef unsigned char[:, :] convol2d_kernel_fast(unsigned char [:, :] image, float [:, :] kernel):
    print('Cython apply kernel function.')
    # set the variable extension types
    cdef int x
    cdef int y
    cdef int w
    cdef int h
    cdef int kh
    cdef int kw
    cdef int i
    cdef int j
    cdef float t
    cdef unsigned char[:, :] imgPadding
    cdef unsigned char[:, :] subImg
    
    resultImg = numpy.zeros_like(image)

    imgPadding = numpy.pad(image, 1, mode='constant')
    
    # grab the image and kernel dimensions
    h = image.shape[0]
    w = image.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # print('Cython size of original image y,x', h, w)
    # print('Cython size of padding image y,x', imgPadding.shape[0], imgPadding.shape[1])
    # print('Cython size of kernel', kh, kw)

    # loop over the image
    for y in range(h):
        for x in range(w):
            subImg = imgPadding[y: y+kh, x: x+kw]
            t = 0
            for j in range(kh):
                for i in range(kw):
                    t += subImg[j, i] * kernel[j, i]
            
            resultImg[y,x] = t
   
    # resultImg = cv2.normalize(resultImg, resultImg, 0,255, cv2.NORM_MINMAX) 
    return resultImg