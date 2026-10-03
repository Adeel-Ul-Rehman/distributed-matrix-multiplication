import numpy as np
import time

MATRIX_SIZE = 4000

A = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE))
B = np.random.randint(0, 10, (MATRIX_SIZE, MATRIX_SIZE))

start_time = time.time()

C = np.dot(A, B)

end_time = time.time()

print("Single System Time:")
print(f"{end_time - start_time:.3f} seconds")

print(C.shape)