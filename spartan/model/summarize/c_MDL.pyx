cdef extern from "math.h":
    double log2(double x)
cdef extern from "stdint.h":
    ctypedef int uint64_t


cpdef double log_comb(int n, int k):
    cdef int i = 0;
    cdef double ret = 0;
    while i < k:
        ret += log2(n-i);
        i += 1;
        ret -= log2(i);
    return ret


cpdef double LN(int n):
    """
    Encode length of an integer z by Rissanen's 1983 Universal code for integers
    """
    if n <= 0:
        return 0;
    cdef double ret = log2(2.865064);
    cdef double i = log2(n);
    while i > 0:
        ret += i;
        i = log2(i);
    return ret


cpdef double LnU(uint64_t n, uint64_t k):
    """
    Minimum encode length of n binary elements with k 1s and (n-k) 0s.
    """
    if n == 0 or k == 0 or k == n:
        return 0
    cpdef double x = -log2(<double> k / n);
    cpdef double y = -log2(<double> (n-k) / n);
    return k * x + (n-k) * y
