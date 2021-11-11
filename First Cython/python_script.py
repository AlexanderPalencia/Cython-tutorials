def test(x):
    y = 0
    for i in range(x):
        y += 1
        print(y)
    return y
# Using Cython