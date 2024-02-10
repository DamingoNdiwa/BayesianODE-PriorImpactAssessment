from scipy.stats import wasserstein_distance
import numpy as np


class WassersteinDistanceCalculator:
    @staticmethod
    def select_columns(arr, columns):
        return arr[:, columns]

    @staticmethod
    def dist_distribu(prefd, pd1):
        w_refd = []
        params = [
            'eta',
            'rho',
            'sigma',
            'I',
            'E',
            'lambda',
            'phi']

        for i in range(len(params)):
            a1 = 'pref_' + params[i]
            'd_' + params[i]
            a = WassersteinDistanceCalculator.select_columns(prefd, i)
            b = WassersteinDistanceCalculator.select_columns(pd1, i)
            w_refd_d1 = wasserstein_distance(a, b)
            w_refd.insert(i, float(w_refd_d1))
            print(
                f'Wasserstein dist bt baseline and dist of interest for {a1} is: {np.round(w_refd_d1,3)}')

        return np.array(w_refd)
