#!python
#cython: language_level=3

cimport cython
cimport numpy as np

import numpy as np

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_slice(np.int64_t[:, ::] sl, int max_):
    cdef:
        np.intp_t start = sl[0, 0]
        np.intp_t stop = sl[0, 1]
        np.intp_t mod_start, mod_stop
        np.int64_t n

    if stop < max_:
        pass

    elif start and start >= max_:
       n = stop - start
       mod_start = start % max_
       if mod_start + n >= max_:
           sl[0, 0] = mod_start
           sl[0, 1] = max_
           sl[1, 0] = 0
           sl[1, 1] = n - (max_-mod_start)

       else:
           sl[0, 0] = mod_start
           sl[0, 1] = mod_start + n

    else:
        mod_stop = stop % max_
        sl[0, 0] = start
        sl[0, 1] = max_
        sl[1, 0] = 0
        sl[1, 1] = mod_stop

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef copy_arr(np.intp_t row, np.int64_t[:, ::] sl, int max_,
                float[::, :] tb, float[::] cp_from, bint compute):

    cdef np.intp_t i, start, stop

    if compute:
        compute_slice(sl, max_)

    start = sl[0, 0]
    stop = sl[0, 1]

    for i in range(start, stop):
        tb[row, i] = cp_from[i - start]  # View entire thing

    if sl[1, 1] != -1:
        if not sl[1, 1] - sl[1, 0] > tb.shape[1]:
            # slices[1, 1] = slices[1, 0] + 100

            for i in range(sl[1, 0], sl[1, 1]):
                tb[row, i] = cp_from[i - sl[1, 0] + stop - start]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_cy(float[:, ::] cy, np.int64_t[::] ind_A, np.int64_t[::] idx, float[::, :] ccf,
                float[:, ::] y, np.int64_t N, np.int64_t nb_, float t):

    cdef np.intp_t m, i

    for m in range(N):
        for i in range(idx[m], idx[m+1]):
            cy[m + nb_, ind_A[i]] *= (1 - 1. / t)

        for i in range(idx[m], idx[m+1]):
            cy[m + nb_, ind_A[i]] += ccf[m + nb_, 0] * y[0, ind_A[i]] / t

    cy_col = cy.shape[1]
    for i in range(nb_):
        for j in range(cy_col):
            cy[i, j] = cy[i, j] * (1 - 1. / t) + (ccf[i, 0] * y[0, j] / t)