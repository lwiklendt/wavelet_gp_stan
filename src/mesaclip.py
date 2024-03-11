import numpy as np
from numba import njit


@njit
def find_extrema(x):
    """
    Finds peaks and troughs in x and stores them in m. Ensures that no two peaks exist without a trough between them
    and visa-versa. Also ensures that find_extrema(m_pos, x) and find_extrema(m_neg, -x) implies that m_pos == -m_neg.
    :param x: input array for which to inspect for peaks and troughs, must of length >= 2.
    :return: int8 array equal length to x, storing values -1 for trough, 1 for peak, and 0 for neither
    """

    # although the word "gradient" or "grad" is used here,
    # it is meant as the gradient sign or direction only rather than the sign and magnitude of the gradient

    m = np.empty(len(x), dtype=np.int8)

    # if negative gradient, then we consider the end a peak, positive gradient is a trough, otherwise neither
    m[0] = int(np.sign(x[0] - x[1]))

    grad_mem = 0
    for i in range(1, len(x) - 1):

        # obtain the direction of the gradient before and after i
        grad_prev = int(np.sign(x[i] - x[i - 1]))
        grad_next = int(np.sign(x[i + 1] - x[i]))

        # carry the last non-zero gradient through if we're in a plateau (unless grad_mem is also 0 from start)
        if grad_prev == 0:
            grad_prev = grad_mem

        # p = grad_prev (could be current or carried over from grad_mem)
        # n = grad_next
        # a = any (can be either -1, 0, or 1, inconsequential which)
        #
        #          can get this            by using this
        #   p   n  ->   m        n-p   p*n   (n-p)*p*n
        #  ----------------------------------------------
        #   0   a  ->   0         a     0        0
        #   a   0  ->   0        -a     0        0
        #  -1  -1  ->   0         0     1        0
        #   1   1  ->   0         0     1        0
        #  -1   1  ->  -1         2    -1       -2
        #   1  -1  ->   1        -2    -1        2

        # m[i] will contain 1 for a peak, -1 for a trough, and 0 if neither, based on the above table
        m[i] = np.sign((grad_next - grad_prev) * grad_prev * grad_next)

        # remember the gradient so that it may be carried forward when we enter a plateau
        if grad_prev != 0:
            grad_mem = grad_prev

    # if positive gradient, then we consider the end a peak, negative gradient is a trough, otherwise neither
    m[-1] = int(np.sign(x[-1] - x[-2]))

    return m


@njit
def mesaclip(x, y, k):
    """
    Clips the peaks of y to plateaus of minimum distance k, where the distance between i and j is x[j] - x[i].
    :param x: non-decreasing input array specifying the x position of each element of y
    :param y: input/output array, heights for clipping
    :param k: input scalar, minimum plateau distance
    """

    # ensure non-decreasing x
    if np.any(np.diff(x) < 0):
        raise RuntimeError('x is not non-decreasing')

    # if entire signal is too short, then clip everything
    if x[-1] - x[0] < k:
        y[:] = np.min(y)
        return

    n = len(y)

    # 1 for peak, -1 for trough, 0 otherwise
    m = find_extrema(y)

    # stack stores active ranges to the left of the current peak, stack size is si, top element is at si - 1
    si = 0
    sa = np.empty(n//2 + 1, dtype=np.int64)  # impossible to have more peaks, and hence ranges, than ceil(n/2)
    sb = np.empty(n//2 + 1, dtype=np.int64)

    # scan for peaks
    i = -1
    while i < n - 1:
        i += 1

        if m[i] == 1:

            # found peak

            # hold an expanding range [a:b+1] where a can decrease and b increase,
            a = i
            b = i

            # expanding the range to the left and right, there are only three ways to break out of this loop:
            #   1) when the range is large enough to span the desired minimum distance: x[b] - x[a] >= k
            #   2) can no longer increase the range because a == 0 and b == n - 1
            #   3) we've reached a trough on the right side of the range
            while True:

                # calculate the range width
                d = x[b] - x[a]

                # if we have sufficient range width
                if d >= k:

                    # clip this range
                    y[a:b + 1] = min(y[a], y[b])

                    # go to next peak
                    i = b
                    break

                # we've reached a left trough that isn't at the start
                if m[a] == -1 and a > 0:

                    # can the range on the top of the stack be combined with the current range?
                    if si > 0 and sb[si - 1] == a:

                        # pop and combine
                        si -= 1
                        t = sa[si]
                        y[t] = min(y[t], y[a])  # retain potential minimum at the combine index
                        a = t

                        # we've just extended the range, so loop around to test against k
                        continue

                    else:
                        # push onto stack
                        sa[si] = a
                        sb[si] = b
                        si += 1

                        # go to next peak
                        i = b
                        break

                # we've reached a right trough that isn't at the end
                if m[b] == -1 and b < n - 1:

                    # push onto stack
                    sa[si] = a
                    sb[si] = b
                    si += 1

                    # go to next peak
                    i = b
                    break

                # left or right step forced by range touching an end
                if a == 0:
                    b += 1
                elif b == n - 1:
                    a -= 1

                # otherwise step towards the larger value
                elif y[a - 1] > y[b + 1]:
                    a -= 1
                else:
                    b += 1

    # clip all remaining ranges
    while si > 0:
        si -= 1
        a = sa[si]
        b = sb[si]
        y[a:b + 1] = min(y[a], y[b])


@njit
def verify_mesaclip(x, y, k):
    n = len(x)
    for a in range(n):
        for b in range(a + 1, n):

            # if width is too short, then check failure conditions
            if x[b] - x[a] < k:

                # failure condtions check for:
                #
                # peak:
                #                  _____
                #            _____/     \_____
                #              ^           ^
                #              a           b
                #
                # stepup at end:
                #                    ______
                #            _______/
                #               ^         ^
                #               a        b=n-1
                #
                # stepdown at start:
                #            _______
                #                   \______
                #            ^         ^
                #           a=0        b

                # peak
                if np.max(y[a:b+1]) != max(y[a], y[b]):
                    return True, (a, b, 0)

                # stepup at the end
                if b == n-1 and y[a] < y[b]:
                    return True, (a, b, 1)

                # stepdown at the start
                if a == 0 and y[a] > y[b]:
                    return True, (a, b, 2)

    return False, (-1, -1, -1)


def test_mesaclip():

    from scipy.ndimage.filters import gaussian_filter1d
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    np.random.seed(1904041351)

    num_tests = 1_000_000
    failed_idxs = []
    for i in range(num_tests):

        num_failed = len(failed_idxs)

        if (i + 1) % (num_tests // 1000) == 0:
            print(f'completed {100 * (i+1)/num_tests}%, failed {num_failed}/{i+1} ({100 * num_failed/(i+1):.2f}%)', flush=True)

        # generate a random signal
        length = np.random.randint(16, 1024 + 1)
        x = np.exp(gaussian_filter1d(np.random.randn(length), sigma=np.exp(np.random.randn())))
        x = np.cumsum(x + np.std(x) * np.abs(np.random.randn()))
        y = gaussian_filter1d(np.cumsum(np.random.randn(length)), sigma=np.exp(np.random.randn()))
        k = 1.1 * np.random.rand()**2 * (x[-1] - x[0])

        # keep original for plot
        yorig = y.copy()

        # run algorithm
        mesaclip(x, y, k)

        # perform verification
        failed, failed_at = verify_mesaclip(x, y, k)
        if failed:
            failed_idxs.append(i)

        if failed:
            fig = plt.figure(figsize=(20, 7))
            locator = ticker.MaxNLocator(min(fig.get_size_inches()[0], length))

            ax = fig.add_subplot(311)
            if failed:
                ax.set_title(f'i={i}  k={k}   failed_at={failed_at}  d={x[failed_at[1]] - x[failed_at[0]]}')
            else:
                ax.set_title(f'i={i}  k={k}   passed')
            ax.plot(range(length - 1), np.diff(x))
            ax.plot(range(length - 1), np.diff(x), ls='none', marker='.', ms=3, alpha=0.5)
            ax.set_xlim(-1, length)
            ax.xaxis.set_major_locator(locator)

            ax = fig.add_subplot(312)
            ax.plot(np.arange(length), yorig, lw=1, alpha=0.3, c='b')
            ax.plot(np.arange(length), yorig, ls='none', marker='.', ms=3, c='b', alpha=0.5)
            ax.plot(np.arange(length), y,     lw=1, alpha=0.5, c=(0.8, 0.4, 0))
            ax.plot(np.arange(length), y,     ls='none', marker='.', ms=3, c=(0.8, 0.4, 0))
            if failed:
                ax.plot(failed_at[0], y[failed_at[0]], c='m', ls='none', marker='o', mfc='none')
                ax.plot(failed_at[1], y[failed_at[1]], c='r', ls='none', marker='o', mfc='none')
            ax.set_xlim(-1, length)
            ax.xaxis.set_major_locator(locator)

            ax = fig.add_subplot(313)
            ax.plot(x, yorig, lw=1, alpha=0.3, c='b')
            ax.plot(x, yorig, ls='none', marker='.', ms=3, c='b', alpha=0.5)
            ax.plot(x, y,     lw=1, alpha=0.5, c=(0.8, 0.4, 0))
            ax.plot(x, y,     ls='none', marker='.', ms=3, c=(0.8, 0.4, 0))
            if failed:
                ax.plot(x[failed_at[0]], y[failed_at[0]], c='m', ls='none', marker='o', mfc='none')
                ax.plot(x[failed_at[1]], y[failed_at[1]], c='r', ls='none', marker='o', mfc='none')
            fig.tight_layout()
            plt.show()

    num_failed = len(failed_idxs)
    num_passed = num_tests - num_failed
    print(f'passed {num_passed}/{num_tests} ({100 * num_passed/num_tests:.2f}%)')
    print(f'failed {num_failed}/{num_tests} ({100 * num_failed/num_tests:.2f}%)')

    print(f'failed indexes: {failed_idxs}')


if __name__ == '__main__':
    test_mesaclip()
