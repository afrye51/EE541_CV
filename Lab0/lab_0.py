import skimage
from skimage import io
import time

n = 1
A = io.imread('Archive/grizzlypeakg.png')
(m1, n1) = A.shape
start_time = time.time()

for useless in range(n):
    for i in range(m1):
        for j in range(n1):
            if A[i, j] <= 10:
                A[i, j] = 0

elapsed_time = (time.time() - start_time) / n * 1000
print(elapsed_time)
