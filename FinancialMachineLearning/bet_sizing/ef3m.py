import sys
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import gaussian_kde


class M2N:

    def __init__(self, moments, epsilon=10**-5, factor=5, n_runs=1, variant=1, max_iter=100_000, num_workers=-1):
        self.epsilon = epsilon
        self. factor = factor
        self.n_runs = n_runs
        self.variant = variant
        self.max_iter = max_iter
        self.num_workers = num_workers
        self.moments = moments
        self.new_moments = [0 for _ in range(5)]
        self.parameters = [0 for _ in range(5)]
        self.error = sum([moments[i]**2 for i in range(len(moments))])

    def fit(self, mu_2):
        p_1 = np.random.uniform(0, 1)
        num_iter = 0
        while True:
            num_iter += 1
            if self.variant == 1:
                parameters_new = self.iter_4(mu_2, p_1)
            elif self.variant == 2:
                parameters_new = self.iter_5(mu_2, p_1)
            else:
                raise ValueError("Value of argument 'variant' must be either 1 or 2.")

            if not parameters_new:
                return None

            parameters = parameters_new.copy()
            self.get_moments(parameters)
            error = sum([(self.moments[i] - self.new_moments[i])**2 for i in range(len(self.new_moments))])
            if error < self.error:
                self.parameters = parameters
                self.error = error

            if abs(p_1 - parameters[4]) < self.epsilon:
                break

            if num_iter > self.max_iter:
                return None

            p_1 = parameters[4]
            mu_2 = parameters[1]

        self.parameters = parameters
        return None

    def get_moments(self, parameters, return_result=False):
        u_1, u_2, s_1, s_2, p_1 = parameters
        p_2 = 1 - p_1
        m_1 = p_1 * u_1 + p_2 * u_2
        m_2 = p_1 * (s_1**2 + u_1**2) + p_2 * (s_2**2 + u_2**2)
        m_3 = p_1 * (3 * s_1**2 * u_1 + u_1**3) + p_2 * (3 * s_2**2 * u_2 + u_2**3)
        m_4 = p_1 * (3 * s_1**4 + 6 * s_1**2 * u_1**2 + u_1**4) + p_2 * (3 * s_2**4 + 6 * s_2**2 * u_2**2 + u_2**4)
        m_5 = p_1 * (15 * s_1**4 * u_1 + 10 * s_1**2 * u_1**3 + u_1**5) + p_2 *\
            (15 * s_2**4 * u_2 + 10 * s_2**2 * u_2**3 + u_2**5)

        if return_result:
            return [m_1, m_2, m_3, m_4, m_5]

        self.new_moments = [m_1, m_2, m_3, m_4, m_5]
        return None

    def iter_4(self, mu_2, p_1):
        m_1, m_2, m_3, m_4 = self.moments[0:4]
        param_list = []
        while True:
            mu_1 = (m_1 - (1 - p_1) * mu_2) / p_1
            if (3 * (1 - p_1) * (mu_2 - mu_1)) == 0:
                break

            sigma_2_squared = ((m_3 + 2 * p_1 * mu_1**3 + (p_1 - 1) * mu_2**3 - 3 * mu_1 * (m_2 + mu_2**2 * (p_1-1))) / (3*(1-p_1)*(mu_2-mu_1)))
            if sigma_2_squared < 0:
                break

            sigma_2 = sigma_2_squared**0.5
            sigma_1_squared = ((m_2 - sigma_2**2 - mu_2**2) / p_1 + sigma_2**2 + mu_2**2 - mu_1**2)

            if sigma_1_squared < 0:
                break
            sigma_1 = sigma_1_squared**0.5
            p_1_deno = (3 * (sigma_1**4 - sigma_2**4) + 6 * (sigma_1**2 * mu_1**2 - sigma_2**2 * mu_2**2) + mu_1**4 -
                        mu_2**4)
            if p_1_deno == 0:
                break

            p_1 = (m_4 - 3*sigma_2**4 - 6*sigma_2**2*mu_2**2 - mu_2**4) / p_1_deno
            if (p_1 < 0) or (p_1 > 1):
                break
            param_list = [mu_1, mu_2, sigma_1, sigma_2, p_1]
            break
        if len(param_list) < 5:
            return []

        return param_list

    def iter_5(self, mu_2, p_1):
        m_1, m_2, m_3, m_4, m_5 = self.moments
        param_list = []
        while True:
            mu_1 = (m_1 - (1 - p_1) * mu_2) / p_1
            if (3 * (1 - p_1) * (mu_2 - mu_1)) == 0:
                break
            sigma_2_squared = ((m_3 + 2 * p_1 * mu_1**3 + (p_1-1) * mu_2**3 - 3 * mu_1 * (m_2 + mu_2**2 * (p_1-1))) /
                               (3*(1-p_1)*(mu_2-mu_1)))
            if sigma_2_squared < 0:
                break
            sigma_2 = sigma_2_squared**0.5
            sigma_1_squared = ((m_2 - sigma_2**2 - mu_2**2)/p_1 + sigma_2**2 + mu_2**2 - mu_1**2)
            if sigma_1_squared < 0:
                break
            sigma_1 = sigma_1_squared**0.5
            if (1 - p_1) < 1e-4:
                break

            a_1_squared = (6 * sigma_2**4 + (m_4 - p_1 * (3 * sigma_1**4 + 6 * sigma_1**2 * mu_1**2 + mu_1**4)) /
                           (1-p_1))
            if a_1_squared < 0:
                break

            a_1 = a_1_squared**0.5
            mu_2_squared = (a_1 - 3 * sigma_2**2)
            if np.iscomplex(mu_2_squared) or mu_2_squared < 0:
                break

            mu_2 = mu_2_squared**0.5
            a_2 = 15 * sigma_1**4 * mu_1 + 10 * sigma_1**2 * mu_1**3 + mu_1**5
            b_2 = 15 * sigma_2**4 * mu_2 + 10 * sigma_2**2 * mu_2**3 + mu_2**5
            if (a_2 - b_2) == 0:
                break

            p_1 = (m_5 - b_2) / (a_2 - b_2)
            if (p_1 < 0) or (p_1 > 1):
                break
            param_list = [mu_1, mu_2, sigma_1, sigma_2, p_1]
            break
        if len(param_list) < 5:
            return []

        return param_list

    def single_fit_loop(self, epsilon=0):
        self.epsilon = epsilon if epsilon != 0 else self.epsilon
        self.parameters = [0 for _ in range(5)]
        self.error = sum([self.moments[i]**2 for i in range(len(self.moments))])

        std_dev = centered_moment(self.moments, 2)**0.5
        mu_2 = [float(i) * self.epsilon * self.factor * std_dev + self.moments[0] for i in range(1, int(1/self.epsilon))]
        err_min = self.error

        d_results = {}
        for mu_2_i in mu_2:
            self.fit(mu_2=mu_2_i)

            if self.error < err_min:
                err_min = self.error
                d_results['mu_1'], d_results['mu_2'], d_results['sigma_1'], d_results['sigma_2'], d_results['p_1'] = \
                    [[p] for p in self.parameters]
                d_results['error'] = [err_min]

        return pd.DataFrame.from_dict(d_results)

    def mp_fit(self):
        num_workers = self.num_workers if self.num_workers > 0 else cpu_count()
        pool = Pool(num_workers)

        output_list = pool.imap_unordered(self.single_fit_loop, [self.epsilon for i in range(self.n_runs)])
        df_list = []
        max_prog_bar_len = 25
        for i, out_i in enumerate(output_list, 1):
            df_list.append(out_i)
            num_fill = int((i/self.n_runs) * max_prog_bar_len)
            prog_bar_string = '|' + num_fill*'#' + (max_prog_bar_len-num_fill)*' ' + '|'
            sys.stderr.write(f'\r{prog_bar_string} Completed {i} of {self.n_runs} fitting rounds.')
        pool.close()
        pool.join()
        df_out = pd.concat(df_list)
        return df_out

def centered_moment(moments, order):
    moment_c = 0
    for j in range(order + 1):
        combin = int(comb(order, j))
        if j == order:
            a_1 = 1
        else:
            a_1 = moments[order - j - 1]
        moment_c += (-1)**j * combin * moments[0]**j * a_1

    return moment_c

def raw_moment(central_moments, dist_mean):
    raw_moments = [dist_mean]
    central_moments = [1] + central_moments
    for n_i in range(2, len(central_moments)):
        moment_n_parts = []
        for k in range(n_i + 1):
            sum_part = comb(n_i, k) * central_moments[k] * dist_mean**(n_i - k)
            moment_n_parts.append(sum_part)
        moment_n = sum(moment_n_parts)
        raw_moments.append(moment_n)
    return raw_moments

def most_likely_parameters(data, ignore_columns = 'error', res = 10000):
    df_results = data.copy()
    if isinstance(ignore_columns, str):
        ignore_columns = [ignore_columns]
    columns = [c for c in df_results.columns if c not in ignore_columns]
    d_results = {}
    for col in columns:
        x_range = np.linspace(df_results[col].min(), df_results[col].max(), num=res)
        kde = gaussian_kde(df_results[col].to_numpy())
        y_kde = kde.evaluate(x_range)
        top_value = round(x_range[np.argmax(y_kde)], 5)
        d_results[col] = top_value

    return d_results
