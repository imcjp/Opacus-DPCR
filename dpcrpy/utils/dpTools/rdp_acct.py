"""
This file contains the implementation of the main class object:  anaRDPacct --- an analytical moment accountant
that keeps track the effects of a hetereogeneous sequence of randomized algorithms using the RDP technique.

In particular it supports amplification of RDP by subsampling without replacement and the amplification of RDP
by poisson sampling, but unfortunately not (yet) together.


* The RDP accountant API is deprecated.
* All functionality of an RDP accountant can be achieved via the Mechanism API.

"""






import numpy as np
from scipy.optimize import minimize_scalar
import sys
sys.path.append('..')

from . import utils, rdp_bank
from .privacy_calibrator import subsample_epsdelta
import scipy
import math

def general_upperbound(func, mm, prob):
    """

    :param func:
    :param mm: alpha in RDP
    :param prob: sample probability
    :return: the upperbound in theorem 1 in 2019 ICML,could be applied for general case(including poisson distribution)
    k_approx = 100 k approximation is applied here
    """
    def cgf(x):
        return (x - 1) * func(x)

    if np.isinf(func(mm)):
        return np.inf
    if mm == 1 or mm == 0:
        return 0

    cur_k = np.minimum(50, mm - 1) # choose small k-approx for general upperbound (here is 50) in case of scipy-accuracy
    log_term_1 = mm * np.log(1 - prob)
    #logBin = utils.get_binom_coeffs(mm)
    log_term_2 = np.log(3) - func(mm) + mm * utils.stable_logsumexp_two(np.log(1 - prob), np.log(prob) + func(mm))
    neg_term_3 = [np.log(scipy.special.comb(mm,l)) + np.log(3) + (mm - l) * np.log(1 - prob) + l * np.log(prob) +
                  utils.stable_log_diff_exp((l - 1) * func(mm), cgf(l))[1] for l in
                  range(3, cur_k + 1)]
    neg_term_4 = np.log(mm*(mm - 1)/2) + 2 * np.log(prob) + (mm - 2) * np.log(
        1 - prob) + utils.stable_log_diff_exp(np.log(3) + func(mm), func(2))[1]
    neg_term_5 = np.log(2) + np.log(prob) + np.log(mm) + (mm - 1) * np.log(1 - prob)
    neg_term_6 = mm * np.log(1 - prob) + np.log(3) - func(mm)
    pos_term = utils.stable_logsumexp([log_term_1, log_term_2])
    neg_term_3.append(neg_term_4)
    neg_term_3.append(neg_term_5)
    neg_term_3.append(neg_term_6)
    neg_term = utils.stable_logsumexp(neg_term_3)
    bound = utils.stable_log_diff_exp(pos_term, neg_term)[1]
    return bound

def fast_subsampled_cgf_upperbound(func, mm, prob, deltas_local):
    # evaulate the fast CGF bound for the subsampled mechanism
    # func evaluates the RDP of the base mechanism
    # mm is alpha.  NOT lambda.
    return np.inf

    if np.isinf(func(mm)):
        return np.inf
    if mm == 1:
        return 0

    secondterm = 2 * np.log(prob) + + np.log(mm) + np.log(mm - 1) - np.log(2) \
                  + np.mininum(np.log(4) + func(2.0) + np.log(1 - np.exp(-func(2.0))),
                               func(2.0) + np.mininum(np.log(2),
                                                      2 * (eps_inf + np.log(1 - np.exp(-eps_inf)))))

    # secondterm = np.minimum(np.minimum((2) * np.log(np.exp(func(np.inf)) - 1)
    #                                    + np.minimum(func(2), np.log(4)),
    #                                    np.log(2) + func(2)),
    #                         np.log(4) + 0.5 * deltas_local[int(2 * np.floor(2 / 2.0)) - 1]
    #                         + 0.5 * deltas_local[int(2 * np.ceil(2 / 2.0)) - 1]
    #                         ) + 2 * np.log(prob) + np.log(mm) + np.log(mm - 1) - np.log(2)

    if mm == 2:
        return utils.stable_logsumexp([0, secondterm])

    # approximate the remaining terms using a geometric series or binomial series

    log_exp_eps_minus_one = func(np.inf) + np.log(1 - np.exp(-func(np.inf)))


    if mm == 3:
        return utils.stable_logsumexp([0, secondterm, (3 * (np.log(prob) + np.log(mm))
                                                       + 2*func(mm)
                                                       + np.minumum(np.log(2),
                                                                    3 * log_exp_eps_minus_one))])

    logratio1 = np.log(prob) + np.log(mm) + func(mm)
    logratio2 = logratio1 + log_exp_eps_minus_one

    s, mag = utils.stable_log_diff_exp(1,logratio1)
    s, mag2 = utils.stable_log_diff_exp(1, (mm-3)*logratio1)
    remaining_terms1 = (np.log(2) + 3 * (np.log(prob) + np.log(mm)) + 2*func(mm)
                        + mag2 - mag)

    s, mag = utils.stable_log_diff_exp(1,logratio2)
    s, mag2 = utils.stable_log_diff_exp(1, (mm-3)*logratio2)

    remaining_terms2 = (3 * (np.log(prob) + np.log(mm) + log_exp_eps_minus_one) + 2*func(mm)
                        + mag2 - mag)

    return utils.stable_logsumexp([0, secondterm, np.minimum(remaining_terms1,remaining_terms2)])

    # logratio = np.minimum(logratio1, logratio2)
    # if logratio1 > logratio2:
    #     coeff = 1
    # else:
    #     coeff = 2
    #
    #
    # if mm == 3:
    #     return utils.stable_logsumexp([0, secondterm, np.log(coeff) + 3 * logratio])
    #
    # # Calculate the sum of the geometric series starting from the third term. This is a total of mm-2 terms.
    # if logratio < 0:
    #     geometric_series_bound = np.log(coeff) + 3 * logratio - np.log(1 - np.exp(logratio)) \
    #                              + np.log(1 - np.exp((mm - 2) * logratio))
    # elif logratio > 0:
    #     geometric_series_bound = np.log(coeff) + 3 * logratio + (mm-2) * logratio - np.log(np.exp(logratio) - 1)
    # else:
    #     geometric_series_bound = np.log(coeff) + np.log(mm - 2)
    #
    # # we will approximate using (1+h)^mm
    # logh1 = np.log(prob) + func(mm - 1)
    #
    # logh2 = logh1 + np.log(np.exp(func(np.inf)) - 1)
    #
    # binomial_series_bound1 = np.log(2) + mm * utils.stable_logsumexp_two(0, logh1)
    # binomial_series_bound2 = mm * utils.stable_logsumexp_two(0, logh2)
    #
    # tmpsign, binomial_series_bound1 \
    #     = utils.stable_sum_signed(True, binomial_series_bound1, False, np.log(2)
    #                               + utils.stable_logsumexp([0, logh1 + np.log(mm), 2 * logh1 + np.log(mm)
    #                                                         + np.log(mm - 1) - np.log(2)]))
    # tmpsign, binomial_series_bound2 \
    #     = utils.stable_sum_signed(True, binomial_series_bound2, False,
    #                               utils.stable_logsumexp([0, logh2 + np.log(mm), 2 * logh2 + np.log(mm)
    #                                                       + np.log(mm - 1) - np.log(2)]))
    #
    # remainder = np.min([geometric_series_bound, binomial_series_bound1, binomial_series_bound2])
    #
    # return utils.stable_logsumexp([0, secondterm, remainder])



def fast_poission_subsampled_cgf_upperbound(func, mm, prob):
    # evaulate the fast CGF bound for the subsampled mechanism
    # func evaluates the RDP of the base mechanism
    # mm is alpha.  NOT lambda.

    if np.isinf(func(mm)):
        return np.inf
    if mm == 1:
        return 0
    # Bound #1:   log [ (1-\gamma + \gamma e^{func(mm)})^mm ]
    bound1  = mm * utils.stable_logsumexp_two(np.log(1-prob), np.log(prob) + func(mm))

    # Bound #2:   log [  (1-gamma)^alpha E  [ 1 +  gamma/(1-gamma) E[p/q]]^mm ]
    #     log[  (1-gamma)^\alpha    { 1 + alpha gamma / (1-gamma)  + gamma^2 /(1-gamma)^2 * alpha(alpha-1) /2 e^eps(2))
    #  + alpha \choose 3 * gamma^3 / (1-gamma)^3  / e^(-2 eps(alpha)) * (1 + gamma /(1-gamma) e^{eps(alpha)}) ^ (alpha - 3) }
    # ]
    if mm >= 3:
        bound2 = utils.stable_logsumexp([mm * np.log(1-prob), (mm-1) * np.log(1-prob) + np.log(mm) + np.log(prob),
                                     (mm-2)*np.log(1-prob) + 2 * np.log(prob) + np.log(mm) + np.log(mm-1) + func(2),
                                     np.log(mm) + np.log(mm-1) + np.log(mm-2) - np.log(3*2) + 3 * np.log(prob)
                                     + (mm-3)*np.log(1-prob) + 2 * func(mm) +
                                     (mm-3) * utils.stable_logsumexp_two(0, np.log(prob) - np.log(1-prob) + func(mm))])
    else:
        bound2 = bound1

    #print('www={} func={} mm={}'.format(np.exp(func(mm))-1),func, mm)
    #print('bound1 ={} bound2 ={}'.format(bound1,bound2))
    return np.minimum(bound1,bound2)

def fast_k_subsample_upperbound(func, mm, prob, k):
    """

     :param func:
     :param mm:
     :param prob: sample probability
     :param k: approximate term
     :return: k-term approximate upper bound in therorem 11 in ICML-19
     """
    def cgf(x):
        return (x - 1) * func(x)

    if np.isinf(func(mm)):
        return np.inf
    if mm == 1:
        return 0
    #logBin = utils.get_binom_coeffs(mm)
    cur_k = np.minimum(k, mm - 1)
    if (2 * cur_k) >= mm:
        exact_term_1 = (mm - 1) * np.log(1 - prob) + np.log(mm * prob - prob + 1)
        exact_term_2 = [np.log(scipy.special.comb(mm,l))  + (mm - l) * np.log(1 - prob) + l * np.log(prob) + cgf(l) for l in
                        range(2, mm + 1)]
        exact_term_2.append(exact_term_1)
        bound = utils.stable_logsumexp(exact_term_2)
        return bound

    s, mag1 = utils.stable_log_diff_exp(0, -func(mm - cur_k))
    new_log_term_1 = np.log(1 - prob) * mm + mag1
    new_log_term_2 = -func(mm - cur_k) + mm * utils.stable_logsumexp_two(np.log(1 - prob),
                                                                         np.log(prob) + func(mm - cur_k))
    new_log_term_3 = [np.log(scipy.special.comb(mm,l)) + (mm - l) * np.log(1 - prob) + l * np.log(prob) +
                      utils.stable_log_diff_exp((l - 1) * func(mm - cur_k), cgf(l))[1] for l in
                      range(2, cur_k + 1)]
    if len(new_log_term_3) > 0:
        new_log_term_3 = utils.stable_logsumexp(new_log_term_3)
    else:
        return utils.stable_logsumexp_two(new_log_term_1, new_log_term_2)
    new_log_term_4 = [np.log(scipy.special.comb(mm,mm-l)) + (mm - l) * np.log(1 - prob) + l * np.log(prob) +
                      utils.stable_log_diff_exp(cgf(l), (l - 1) * func(mm - cur_k))[1] for l in
                      range(mm - cur_k + 1, mm + 1)]
    new_log_term_4.append(new_log_term_1)
    new_log_term_4.append(new_log_term_2)
    new_log_term_4 = utils.stable_logsumexp(new_log_term_4)
    s, new_log_term_5 = utils.stable_log_diff_exp(new_log_term_4, new_log_term_3)
    new_bound = new_log_term_5
    return new_bound


class anaRDPacct:
    """A class that keeps track of the analytical expression of the RDP --- 1/(alpha-1)*CGF of the privacy loss R.V."""
    def __init__(self, m=100, tol=0.1, m_max=500, m_lin_max=10000, approx = False, verbose=False):
        # m_max indicates the number that we calculate binomial coefficients exactly up to.
        # beyond that we use Stirling approximation.

        # ------ Class Attributes -----------
        self.m = m # default number of binomial coefficients to precompute
        self.m_max = m_max # An upper bound of the quadratic dependence
        self.m_lin_max = m_lin_max # An upper bound of the linear dependence.
        self.verbose = verbose
        self.approx = approx # If true, use the fast k-term approximation
        self.lambs = np.linspace(1, self.m, self.m).astype(int) # Corresponds to \alpha = 2,3,4,5,.... for RDP

        self.alphas = np.linspace(1, self.m, self.m).astype(int)
        self.RDPs_int = np.zeros_like(self.alphas, float)

        self.n=0
        self.RDPs = [] # analytical CGFs
        self.coeffs = []
        self.RDP_inf = .0 # This is effectively for pure DP.
        self.logBinomC = utils.get_binom_coeffs(self.m + 1)  # The logBinomC is only needed for subsampling mechanisms.
        self.idxhash = {} # save the index of previously used algorithms
        self.cache = {} # dictionary to save results from previously seen algorithms
        self.deltas_cache = {} # dictionary to save results of all discrete derivative path
        self.evalRDP = lambda x: 0
        self.flag = True # a flag indicating whether evalCGF is out of date
        self.flag_subsample = False # a flag to indicate whether we need to expand the logBinomC.
        self.tol = tol


    # ---------- Methods ------------
    def build_zeroth_oracle(self):
        self.evalRDP = lambda x:  sum([c * item(x) for (c, item) in zip(self.coeffs, self.RDPs)])

    def plot_rdp(self):
        if not self.flag:
            self.build_zeroth_oracle()
            self.flag = True

        import matplotlib.pyplot as plt
        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        x = range(0,self.m,1)
        y = [self.evalRDP(item) for item in x]
        plt.loglog(x, y)
        plt.show()


    def plot_cgf_int(self):
        import matplotlib.pyplot as plt
        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(self.alphas, self.RDPs_int)
        plt.xlabel(r'$\lambda$')
        plt.ylabel('CGF')
        plt.show()

    def plot_rdp_int(self):
        import matplotlib.pyplot as plt
        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.loglog(self.alphas, self.RDPs_int)
        if not self.flag:
            self.build_zeroth_oracle()
            self.flag = True
        x = range(1,self.m_lin_max,1)
        y = [self.evalRDP(item) for item in x]
        plt.loglog(x, y)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'RDP $\epsilon$')
        plt.show()


    def get_rdp(self,alphas):
        # alphas is a numpy array or a list of numbers
        # we will return a numpy array of the corresponding RDP
        if not self.flag:
            self.build_zeroth_oracle()
            self.flag = True
        alphas = np.array(alphas)
        assert(np.all(alphas >= 1))
        rdp_list = []
        for alpha in alphas:
            rdp_list.append(self.evalRDP(alpha))

        return np.array(rdp_list)

    def get_eps(self, delta): # minimize over \lambda
        if not self.flag:
            self.build_zeroth_oracle()
            self.flag = True

        if delta<0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return self.RDP_inf
        else:
            def fun(x): # the input the RDP's \alpha
                if x <= 1:
                    return np.inf
                else:
                    return np.log(1 / delta)/(x-1) + self.evalRDP(x)

            def fun_int(i): # the input is RDP's \alpha in integer
                if i <= 1 | i >= len(self.RDPs_int):
                    return np.inf
                else:
                    return np.log(1 / delta) / (i-1) + self.RDPs_int[i - 1]


            # When do we have computational constraints?
            # Only when we have subsampled items.

            # First check if the forward difference is positive at self.m, or if it is infinite
            while (self.m<self.m_max) and (not np.isposinf(fun(self.m))) and (fun_int(self.m-1)-fun_int(self.m-2) < 0):
                # If so, double m, expand logBimomC until the forward difference is positive


                if self.flag_subsample:

                    # The following line is m^2 time.
                    self.logBinomC = utils.get_binom_coeffs(self.m*2+1)

                    # Update deltas_caches
                    for key, val in self.deltas_cache.items():
                        if type(key) is tuple:
                            func_tmp = key[0]
                        else:
                            func_tmp = key
                        cgf = lambda x:  x*func_tmp(x+1)
                        deltas,signs_deltas = utils.get_forward_diffs(cgf,self.m*2)

                        self.deltas_cache[key] = [deltas, signs_deltas]

                new_alphas = range(self.m + 1, self.m * 2 + 1, 1)
                self.alphas = np.concatenate((self.alphas, np.array(new_alphas)))  # array of integers
                self.m = self.m * 2

            mm = np.max(self.alphas)

            rdp_int_new = np.zeros_like(self.alphas, float)

            for key,val in self.cache.items():
                idx = self.idxhash[key]
                rdp = self.RDPs[idx]
                newarray = np.zeros_like(self.alphas, float)
                for j in range(2,mm+1,1):
                    newarray[j-1] = rdp(1.0*j)
                newarray[0]=newarray[1]
                coeff = self.coeffs[idx]
                rdp_int_new += newarray * coeff
                self.cache[key] = newarray

            self.RDPs_int = rdp_int_new

            # # update the integer CGF and the cache for each function
            # rdp_int_new = np.zeros_like(self.RDPs_int)
            # for key,val in self.cache.items():
            #     idx = self.idxhash[key]
            #     rdp = self.RDPs[idx]
            #     newarray = np.zeros_like(self.RDPs_int)
            #     for j in range(self.m):
            #         newarray[j] = rdp(1.0*(j+self.m+1))
            #
            #     coeff = self.coeffs[idx]
            #     rdp_int_new += newarray * coeff
            #     self.cache[key] = np.concatenate((val, newarray))
            #
            # # update the corresponding quantities
            # self.RDPs_int = np.concatenate((self.RDPs_int, rdp_int_new))

            #self.m = self.m*2

            bestint = np.argmin(np.log(1 / delta)/(self.alphas[1:]-1) + self.RDPs_int[1:]) + 1

            if bestint == self.m-1:
                if self.verbose:
                    print('Warning: Reach quadratic upper bound: m_max.')
                # In this case, we matches the maximum qudaratic upper bound
                # Fix it by calling O(1) upper bounds and do logarithmic search
                cur = fun(bestint)
                while (not np.isposinf(cur)) and fun(bestint-1)-fun(bestint-2) < -1e-8:
                    bestint = bestint*2
                    cur = fun(bestint)
                    if bestint > self.m_lin_max and self.approx ==True:
                        print('Warning: Reach linear upper bound: m_lin_max.')
                        return cur

                results = minimize_scalar(fun, method='Bounded', bounds=[self.m-1, bestint + 2],
                                          options={'disp': False})
                if results.success:
                    return results.fun
                else:
                    raise RuntimeError(f"Optimal RDP order not found: {results.message}")
                #return fun(bestint)

            if bestint == 0:
                if self.verbose:
                    print('Warning: Smallest alpha = 1.')

            # find the best integer alpha.
            bestalpha = self.alphas[bestint]

            results = minimize_scalar(fun,  method='Bounded',bounds=[bestalpha-1, bestalpha+1],
                                      options={'disp':False})
            # the while loop above ensures that bestint+2 is at most m, and also bestint is at least 0.
            if results.success:
                return results.fun
            else:
                # There are cases when certain \delta is not feasible.
                # For example, let p and q be uniform the privacy R.V. is either 0 or \infty and unless all \infty
                # events are taken cared of by \delta, \epsilon cannot be < \infty
                return np.inf

    def compose_mechanism(self, func, coeff=1.0):
        self.flag = False
        if func in self.idxhash:
            self.coeffs[self.idxhash[func]] += coeff
            # also update the integer CGFs
            self.RDPs_int += self.cache[func] * coeff

        else:
            # book keeping
            self.idxhash[func] = self.n
            self.n += 1
            self.coeffs.append(coeff)
            # update the analytical
            self.RDPs.append(func)

            # also update the integer results
            if func in self.cache:
                tmp = self.cache[func]
            else:
                tmp = np.zeros_like(self.RDPs_int, float)
                for i in range(self.m):
                    tmp[i] = func(i+1)
                self.cache[func] = tmp  # save in cache
            self.RDPs_int += tmp * coeff

        self.RDP_inf += func(np.inf) * coeff
    #795010
    #imple 100
    def compose_subsampled_mechanism(self, func, prob, coeff=1.0, improved_bound_flag=False):
        """
            # This function is for subsample without replacements
        :param func:  RDP function of the mechanism before amplification by sampling
        :param prob:  proportion of the data to sample
        :param coeff: number of times the subsampled mechanism is being composed.
        :param improved_bound_flag:
            - If True, then it uses Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf
            - If False (default value), it uses Theorem 9 of https://arxiv.org/pdf/1808.00087.pdf
            To qualify for the improved bound, the mechanism needs to have a pair of neighboring
            datasets that is worst for all Renyi-divergence and Pearson-Vajda divergence;
            Also, the RDP bound needs to be tight (see Definition 26 from the same paper).
            Gaussian mechanism, Laplace mechanism and many others satisfy this condition.

        :return:  nothing  (updates to the RDP accountant's attribute)
        """

        # (find a random subset of proportion prob)
        self.flag = False
        self.flag_subsample = True
        if (func, prob) in self.idxhash:
            idx = self.idxhash[(func, prob)]
            # update the coefficients of each function
            self.coeffs[idx] += coeff
            # also update the integer CGFs
            self.RDPs_int += self.cache[(func, prob)] * coeff
        else:
            def cgf(x):
                return x * func(x+1)

            if not improved_bound_flag:
                def subsample_func_int(x):
                    # output the cgf of the subsampled mechanism
                    mm = int(x)
                    eps_inf = func(np.inf)

                    moments_two = 2 * np.log(prob) + utils.logcomb(mm,2) \
                                  + np.minimum(np.log(4) + func(2.0) + np.log(1-np.exp(-func(2.0))),
                                               func(2.0) + np.minimum(np.log(2),
                                                            2 * (eps_inf+np.log(1-np.exp(-eps_inf)))))
                    moment_bound = lambda j: np.minimum(j * (eps_inf + np.log(1-np.exp(-eps_inf))),
                                                        np.log(2)) + cgf(j - 1) \
                                             + j * np.log(prob) + utils.logcomb(mm, j)
                    moments = [moment_bound(j) for j in range(3, mm + 1, 1)]
                    return np.minimum((x-1)*func(x), utils.stable_logsumexp([0,moments_two] + moments))
            else:
                # we need forward differences of exp(cgf)
                # The following line is the numerically stable way of implementing it.
                # The output is in polar form with logarithmic magnitude
                deltas, signs_deltas = utils.get_forward_diffs(cgf, self.m)

                #deltas1, signs_deltas1 = get_forward_diffs_direct(func, self.m)

                #tmp = deltas-deltas1

                self.deltas_cache[(func,prob)] = [deltas,signs_deltas]

                def subsample_func_int(x):
                    # This function evaluates teh CGF at alpha = x, i.e., lamb =  x- 1
                    deltas_local, signs_deltas_local = self.deltas_cache[(func,prob)]
                    if np.isinf(func(x)):
                        return np.inf

                    mm = int(x)
                    eps_inf = func(np.inf)

                    moments_two = 2 * np.log(prob) + utils.logcomb(mm, 2) \
                                  + np.minimum(
                        np.log(4) + func(2.0) + np.log(1 - np.exp(-func(2.0))),
                        func(2.0) + np.minimum(np.log(2),
                                               2 * (eps_inf + np.log(1 - np.exp(-eps_inf)))))

                    moment_bound = lambda j: np.minimum(np.log(4) + 0.5*deltas_local[int(2*np.floor(j/2.0))-1]
                                                        + 0.5*deltas_local[int(2*np.ceil(j/2.0))-1],
                                                        np.minimum(j * (eps_inf + np.log(1 - np.exp(-eps_inf))),
                                                                   np.log(2))
                                                        + cgf(j - 1)) \
                                             + j * np.log(prob) + utils.logcomb(mm, j)

                    moment_bound_linear = lambda j: np.minimum(j * (eps_inf + np.log(1-np.exp(-eps_inf))),
                                                        np.log(2)) + cgf(j - 1) \
                                             + j * np.log(prob) + utils.logcomb(mm, j)

                    fastupperbound = fast_subsampled_cgf_upperbound(func, mm, prob, deltas_local)

                    if mm <= self.alphas[-1]: # compute the bound exactly. Requires book keeping of O(x^2)
                        #
                        # moments = [ np.minimum(np.minimum((j)*np.log(np.exp(func(np.inf))-1) + np.minimum(cgf(j-1),np.log(4)),
                        #                                   np.log(2) + cgf(j-1)),
                        #                        np.log(4) + 0.5*deltas_local[int(2*np.floor(j/2.0))-1]
                        #                        + 0.5*deltas_local[int(2*np.ceil(j/2.0))-1]) + j*np.log(prob)
                        #             +self.logBinomC[int(mm), j] for j in range(2,int(mm+1),1)]
                        moments = [moment_bound(j) for j in range(3, mm + 1, 1)]

                        return np.minimum(fastupperbound, utils.stable_logsumexp([0, moments_two] + moments))
                    elif mm <= self.m_lin_max:  # compute the bound with stirling approximation. Everything is O(x) now.
                        # moment_bound = lambda j: np.minimum(j * np.log(np.exp(func(np.inf)) - 1)
                        #                                     + np.minimum(cgf(j - 1), np.log(4)), np.log(2)
                        #                                     + cgf(j - 1)) + j * np.log(prob) + utils.logcomb(mm, j)
                        # moments = [moment_bound(j) for j in range(2,mm+1,1)]


                        moments = [moment_bound_linear(j) for j in range(3, mm + 1, 1)]

                        return np.minimum(fastupperbound, utils.stable_logsumexp([0, moments_two] + moments))
                    else: # Compute the O(1) upper bound
                        return fastupperbound



            def subsample_func(x):
                # This function returns the RDP at alpha = x
                # RDP with the linear interpolation upper bound of the CGF

                epsinf, tmp = subsample_epsdelta(func(np.inf),0,prob)

                if np.isinf(x):
                    return epsinf
                if prob == 1.0:
                    return func(x)

                if (x >= 1.0) and (x <= 2.0):
                    return np.minimum(epsinf, subsample_func_int(2.0) / (2.0-1))
                if np.equal(np.mod(x, 1), 0):
                    return np.minimum(epsinf, subsample_func_int(x) / (x-1) )
                xc = math.ceil(x)
                xf = math.floor(x)
                return np.min(
                    [epsinf,func(x),
                     ((x-xf)*subsample_func_int(xc) + (1-(x-xf))*subsample_func_int(xf)) / (x-1)]
                )


            # book keeping
            self.idxhash[(func, prob)] = self.n # save the index
            self.n += 1 # increment the number of unique mechanisms
            self.coeffs.append(coeff) # Update the coefficient
            self.RDPs.append(subsample_func) # update the analytical functions

            # also update the integer results up to m_max.
            if (func,prob) in self.cache:
                results = self.cache[(func,prob)]
            else:
                results = np.zeros_like(self.RDPs_int, float)
                # m = np.max(self.lambs)
                mm = np.max(self.alphas)
                for alpha in range(2, mm+1):
                    results[alpha-1] = subsample_func(alpha)
                results[0] = results[1] # Provide the trivial upper bound of RDP at alpha = 1 --- the KL privacy.
                self.cache[(func,prob)] = results # save in cache

            self.RDPs_int += results * coeff
        # update the pure DP
        eps, delta = subsample_epsdelta(func(np.inf), 0, prob)
        self.RDP_inf += eps * coeff

    def compose_subsampled_mechanisms_lowerbound(self, func, prob, coeff=1.0):
        self.compose_poisson_subsampled_mechanisms(func, prob, coeff=coeff)

    def compose_poisson_subsampled_mechanisms(self, func, prob, coeff=1.0):
        # This function implements the lower bound for subsampled RDP.
        # It is also the exact formula of poission_subsampled RDP for many mechanisms including Gaussian mech.
        #
        # At the moment, we do not support mixing poisson subsampling and standard subsampling.
        # TODO: modify the caching identifies so that we can distinguish different types of subsampling
        #
        self.flag = False
        self.flag_subsample = True
        if (func, prob) in self.idxhash:
            idx = self.idxhash[(func, prob)] # TODO: this is really where it needs to be changed.
            # update the coefficients of each function
            self.coeffs[idx] += coeff
            # also update the integer CGFs
            self.RDPs_int += self.cache[(func, prob)] * coeff
        else: # compute an easy to compute upper bound of it.

            def cgf(x):
                return x * func(x+1)

            def subsample_func_int(x):
                # This function evaluates teh CGF at alpha = x, i.e., lamb =  x- 1

                if np.isinf(func(x)):
                    return np.inf

                mm = int(x)
                #
                fastbound = fast_poission_subsampled_cgf_upperbound(func, mm, prob)

                k = self.alphas[-1]
                fastbound_k = fast_k_subsample_upperbound(func, mm, prob,k)
                if self.approx == True:
                    return fastbound_k
                #fastbound = min(fastbound, fastbound_k)
                if x <= self.alphas[-1]: # compute the bound exactly.
                    moments = [cgf(j-1) +j*np.log(prob) + (mm-j) * np.log(1-prob)
                               + self.logBinomC[mm, j] for j in range(2,mm+1,1)]

                    return utils.stable_logsumexp([(mm-1)*np.log(1-prob)+np.log(1+(mm-1)*prob)]+moments)
                elif mm <= self.m_lin_max:
                    moments = [cgf(j-1) +j*np.log(prob) + (mm-j) * np.log(1-prob)
                               + utils.logcomb(mm,j) for j in range(2,mm+1,1)]
                    return utils.stable_logsumexp([(mm-1)*np.log(1-prob)+np.log(1+(mm-1)*prob)] + moments)
                else:
                    return fastbound

            def subsample_func(x): # linear interpolation upper bound
                # This function implements the RDP at alpha = x

                if np.isinf(func(x)):
                    return np.inf
                if prob == 1.0:
                    return func(x)
                if prob == 0:
                    return 0

                epsinf, tmp = subsample_epsdelta(func(np.inf),0,prob)

                if np.isinf(x):
                    return epsinf
                if (x >= 1.0) and (x <= 2.0):
                    return np.minimum(epsinf, subsample_func_int(2.0) / (2.0-1))
                if np.equal(np.mod(x, 1), 0):
                    return np.minimum(epsinf, subsample_func_int(x) / (x-1) )
                xc = math.ceil(x)
                xf = math.floor(x)
                return np.minimum(
                    epsinf,
                    ((x-xf)*subsample_func_int(xc) + (1-(x-xf))*subsample_func_int(xf)) / (x-1)
                )

            # book keeping
            self.idxhash[(func, prob)] = self.n # save the index
            self.n += 1 # increment the number of unique mechanisms
            self.coeffs.append(coeff) # Update the coefficient
            self.RDPs.append(subsample_func) # update the analytical functions

            # also update the integer results, with a vectorized computation.
            # TODO: pre-computing subsampled RDP for integers is error-prone (implement the same thing twice)
            # TODO: and its benefits are not clear. We should consider removing it and simply call the lambda function.
            #
            if (func,prob) in self.cache:
                results = self.cache[(func,prob)]
            else:
                results = np.zeros_like(self.RDPs_int, float)
                mm = np.max(self.alphas)  # evaluate the RDP up to order mm
                jvec = np.arange(2, mm + 1)
                logterm3plus = np.zeros_like(results)  # This saves everything from j=2 to j = m+1
                for j in jvec:
                    logterm3plus[j-2] = cgf(j-1) + j * np.log(prob)  #- np.log(1-prob))

                for alpha in range(2, mm+1):
                    if np.isinf(logterm3plus[alpha-1]):
                        results[alpha-1] = np.inf
                    else:
                        tmp = utils.stable_logsumexp(logterm3plus[0:alpha-1] + self.logBinomC[alpha , 2:(alpha + 1)]
                                               + (alpha+1-jvec[0:alpha-1])*np.log(1-prob))
                        results[alpha-1] = utils.stable_logsumexp_two((alpha-1)*np.log(1-prob)
                                                                    + np.log(1+(alpha-1)*prob), tmp) / (1.0*alpha-1)

                results[0] = results[1]  # Provide the trivial upper bound of RDP at alpha = 1 --- the KL privacy.
                self.cache[(func,prob)] = results # save in cache
            self.RDPs_int += results * coeff
        # update the pure DP tracker
        eps, delta = subsample_epsdelta(func(np.inf), 0, prob)
        self.RDP_inf += eps * coeff


    def compose_poisson_subsampled_mechanisms1(self, func, prob, coeff=1.0):
        # This function implements the general amplification bounds for Poisson sampling.
        # No additional assumptions are needed.

        # At the moment, we do not support mixing poisson subsampling and standard subsampling.
        #
        self.flag = False
        self.flag_subsample = True
        if (func, prob) in self.idxhash:
            idx = self.idxhash[(func, prob)]
            # update the coefficients of each function
            self.coeffs[idx] += coeff
            # also update the integer CGFs
            self.RDPs_int += self.cache[(func, prob)] * coeff
        else: # compute an easy to compute upper bound of it.

            cgf = lambda x:  x*func(x+1)

            def subsample_func_int(x):
                # This function evaluates the CGF at alpha = x, i.e., lamb =  x- 1
                if np.isinf(func(x)):
                    return np.inf
                if prob == 1.0:
                    return func(x)

                mm = int(x)

                fastbound = fast_poission_subsampled_cgf_upperbound(func, mm, prob)

                if x <= self.alphas[-1]: # compute the bound exactly.
                    moments = [cgf(1) + 2*np.log(prob) + (mm-2) * np.log(1 - prob) + self.logBinomC[mm, 2]]
                    moments = moments + [cgf(j-1+1) +j*np.log(prob) + (mm-j) * np.log(1 - prob)
                               + self.logBinomC[mm, j] for j in range(3,mm+1,1)]

                    return utils.stable_logsumexp([(mm-1)*np.log(1-prob)+np.log(1+(mm-1)*prob)]+moments)
                elif mm <= self.m_lin_max:
                    moments = [cgf(1) + 2*np.log(prob) + (mm-2) * np.log(1 - prob) + utils.logcomb(mm, 2)]
                    moments = moments + [cgf(j-1+1) +j*np.log(prob) + (mm-j) * np.log(1 - prob)
                               + utils.logcomb(mm, j) for j in range(3,mm+1,1)]
                    return utils.stable_logsumexp([(mm-1)*np.log(1-prob)+np.log(1+(mm-1)*prob)]+moments)
                else:
                    return fastbound


            def subsample_func(x): # linear interpolation upper bound
                if prob == 0:
                    return 0
                if prob == 1.0:
                    return func(x)
                epsinf, tmp = subsample_epsdelta(func(np.inf),0,prob)

                if np.isinf(x):
                    return epsinf
                if (x >= 1.0) and (x <= 2.0):
                    return np.minimum(epsinf, subsample_func_int(2.0) / (2.0-1))
                if np.equal(np.mod(x, 1), 0):
                    return np.minimum(epsinf, subsample_func_int(x) / (x-1) )
                xc = math.ceil(x)
                xf = math.floor(x)
                return np.minimum(
                    epsinf,
                    ((x-xf)*subsample_func_int(xc) + (1-(x-xf))*subsample_func_int(xf)) / (x-1)
                )

            # book keeping
            self.idxhash[(func, prob)] = self.n # save the index
            self.n += 1 # increment the number of unique mechanisms
            self.coeffs.append(coeff) # Update the coefficient
            self.RDPs.append(subsample_func) # update the analytical functions

            # also update the integer results
            if (func,prob) in self.cache:
                results = self.cache[(func,prob)]
            else:
                results = np.zeros_like(self.RDPs_int, float)
                mm = np.max(self.alphas)  # evaluate the RDP up to order mm

                for alpha in range(2, mm+1):
                    results[alpha-1] = subsample_func_int(alpha)
                results[0] = results[1]  # Provide the trivial upper bound of RDP at alpha = 1 --- the KL privacy.
                self.cache[(func,prob)] = results # save in cache
            self.RDPs_int += results * coeff
        # update the pure DP tracker
        eps, delta = subsample_epsdelta(func(np.inf), 0, prob)
        self.RDP_inf += eps * coeff


# TODO: 1. Modularize the several Poission sampling versions.  2. Support both sampling schemes together.
