# import python_iterate_over_img
import iterate_over_img
import cv2
import numpy as np
import time

image = cv2.imread("./imgs/test_train.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# start = time.time()
# a = python_iterate_over_img.threshold_slow(5,image)
# end =  time.time()

# py_time = end - start
# print("Python time to run a image is {}".format(py_time))


start = time.time()
b = iterate_over_img.threshold_fast(image)
end =  time.time()

cy_time = end - start
print("Cython time to run a image is {}".format(cy_time))

npB = (np.asarray(b))
print(type(npB))
print(npB.max())
print(npB.shape)
# print("Speedup = {}".format(py_time / cy_time))



# # print(b)
# # print(b.shape)
# # print(np.asarray(b))

# print('Results Python and Cython are the same?? ', np.array_equal(a,b))