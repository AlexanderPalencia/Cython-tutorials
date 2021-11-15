import timeit

py = timeit.timeit('test_python.test(10)', setup='import test_python', number=100)

cy = timeit.timeit('test_cython.test(10)', setup='import test_cython', number=100)

print(cy, py)
print('Cython is {} faster'.format(py/cy))


import test_python
import test_cython
import time

number = 90

start = time.time()
a = test_python.test(number)
end =  time.time()

py_time = end - start
print("Python time = {}, result is {}".format(py_time, a))


start = time.time()
b = test_cython.test(number)
end =  time.time()

cy_time = end - start
print("Cython time = {}, result is {}".format(cy_time, b))

print("Speedup = {}".format(py_time / cy_time))

