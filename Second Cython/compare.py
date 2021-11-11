import timeit

cy = timeit.timeit('test_python.test(10)', setup='import test_python', number=100)

py = timeit.timeit('test_cython.test(10)', setup='import test_cython', number=100)

print(cy, py)
print('Cython is {} faster'.format(py/cy))