import cython_script
import python_script
import time

number = 100000

start = time.time()
a = python_script.test(number)
end =  time.time()

py_time = end - start
print("Python time = {}, result".format(py_time))


start = time.time()
b = cython_script.test(number)
end =  time.time()

cy_time = end - start
print("Cython time = {}, result".format(cy_time))

print("Speedup = {}".format(py_time / cy_time))

