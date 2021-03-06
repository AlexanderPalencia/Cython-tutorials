# Manuel Alexander Palencia Gutierrez
# https://cython.readthedocs.io/en/latest/src/quickstart/build.html
# https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/?_ga=2.248107335.1553146457.1636182283-1879079528.1630216350

# https://www.pyimagesearch.com/2019/09/09/multiprocessing-with-opencv-and-python/
# https://www.pyimagesearch.com/2017/08/28/fast-optimized-for-pixel-loops-with-opencv-and-python/
# https://www.youtube.com/watch?v=mXuEoqK4bEc
import numpy as np

v = np.array([[3,4],[32,5]])
print(v)

print(np.pad(v, 1, mode='constant'))