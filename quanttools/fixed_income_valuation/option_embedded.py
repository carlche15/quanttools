import numpy as np
from quanttools.fixed_income_valuation import main_settings

class Swaption_G2:
    """

    Price a swaption by a numerical integration of a closed-form function. Function and data members within this class
    are small components of this massive integration. For future reference, each function member is assigned a "function level",
    the smaller the number, the higer the level.
    """

    def __init__(self, maturity, tenor, strike, ini_curve, spread_curve=None, type='payer',
                 black_price=None, implied_vol=None):
        import numpy as np
        self.maturity = maturity  # maturity of swaption, quoted in year, i.e. maturity=10; float or int
        self.pmtdates = maturity + np.arange(1,
                                             1 + tenor)  # interest rate swap dates, quoted in a series of years, i.e. maturity+1, maturity+2.....
        self.tenor = tenor  # number of swaps in the swaption
        self.nominal = 100  # nominal amount of swaption;
        self.index_multipler = 12  # index used to retrieve data on monthly curves;  CAN NOT BE CHANGED;
        self.strike = strike  # dollar strike price of  swaption
        self.ini_curve = ini_curve  # OIS curve at the valuation date
        self.adj = 1.0  # price adjust for dual curve
        self.spread_curve = spread_curve  # the ratio of LIBOR and OIS curve at the valuation date
        self.black_price = black_price  # black price of swaption, not being called by any data/function member
        self.implied_vol = implied_vol  # not being called by any data/function member
        self.type = type  # "receiver" or "payer"; it DOES matter if swaption is not at the money

        """
        G2 parameters and dependent parameters; the values are initialized with -999.
        """
        self.param = {"a": -999.,
                      "b": -999.,
                      "sigma": -999.,
                      "eta": -999.,
                      "rho": -999.,
                      "miu_x": -999.,
                      "miu_y": -999.,
                      "sigma_x": -999.,
                      "sigma_y": -999.,
                      "rho_xy": -999.,
                      }
        self.dual_curve_adj()
        self.set_payer_receiver(type)
        self.MCprice = None
        self.black_price = self.calc_price_lognorm(self.implied_vol)

    def set_payer_receiver(self, type):
        """

        :param type:
        :return:
        """
        "level 0"
        if type == 'receiver':
            self.omega = -1
        elif type == 'payer':
            self.omega = 1
        else:
            raise IndexError("type could only be receiver or payer")

    def dual_curve_adj(self):
        """
        Adjust Fixed leg payment rates in order to fit the dual curve valuation in the single-curve
        swaption framework. Upon called, self.ci and self.adj(see ___init___ defination) are adjusted.
        The spread between LIBOR and OIS is measured as their ratio Ds (spread curve)=Dl/Dd
        Reference:
        [1] Multiple-Curve Valuation With One-Factor Hull White Model, Jun Zhu, 2012

        Parameters:
        ----------------------
        self.spread_curve: 1d array, current monthly spread curve; e.g.[0.9999.0.9987....]
        self.ini_curve: 1d array, current monthly discount (OIS) curve; e.g.[0.999,0.987,.0975......]
        self.maturity:  maturity of swaption, quoted in year; i.e. maturity=10; float or int
        self.tenor: number of (annual)swaps in the swaption; for 10-y swaption tenor=10
        self.ci: fixed leg interest rate with last rate adjusted i.e. ci=[0.02,0.02,....0.02, 1+0.02]

        Returns:
        ----------------------
        0 (not being used) if self.spread_curve argument is None, i.e. single curve scenario
        2 if if self.spread_curve argument is not None, i.e. double curve scenario; and self.ci, self.adj are adjusted
        """
        "level 0"
        if self.spread_curve is None or main_settings.curve_num == 1:
            ci = np.repeat(self.strike, self.tenor)
            ci[-1] += 1
            self.ci = np.array(ci)
            return 0

        elif main_settings.spread_type == 'spread' and main_settings.curve_num == 2:

            ois_curve = self.ini_curve
            libor_curve = (self.ini_curve * self.spread_curve)
            time_yr = np.arange(len(ois_curve)) / 12.
            ois_spot_rate = (1. / ois_curve) ** (1. / time_yr) - 1  # getting annual spot rate
            libor_spot_rate = (1. / libor_curve) ** (1. / time_yr) - 1
            spot_rate_spread = libor_spot_rate - ois_spot_rate

            ci = np.repeat(self.strike, self.tenor)
            ci[-1] += 1
            self.ci = np.array(ci)
            self.ci_orig = self.ci

            # s_is = np.array([spot_rate_spread[int(12 * (ti - 1))] for ti in self.maturity + np.arange(1, self.tenor + 1)])
            # #
            s_is = np.array([(libor_curve[int(12 * (ti - 1))] / libor_curve[int(12 * (ti))] - 1)
                             - (ois_curve[int(12 * (ti - 1))] / ois_curve[int(12 * (ti))] - 1)
                             for ti in self.maturity + np.arange(1, self.tenor + 1)])

            mi = (self.ci - s_is)
            self.ci = mi
            self.adj = 1.0
            return 2
        elif main_settings.spread_type == 'ratio' and main_settings.curve_num == 2:
            ci = np.repeat(self.strike, self.tenor)
            ci[-1] += 1
            self.ci = np.array(ci)
            b_is = np.array([self.spread_curve[int(12 * (ti - 1))] / self.spread_curve[int(12 * (ti))] for ti in
                             self.maturity + np.arange(1, self.tenor + 1)])
            b_n = b_is[-1]
            b_bar = np.mean(b_is)
            Ld_0_1 = (1. / self.ini_curve[12] - 1)
            g_is = (b_is - 1) + (b_is - b_bar) * Ld_0_1
            g_is[-1] = b_n - b_bar + (b_n - b_bar) * Ld_0_1
            mi = (self.ci - g_is) / b_bar
            self.ci = mi
            self.adj = b_bar
            return 2

        # self.set_parameter(ini_prarm)

    def set_parameter(self, G2_param):

        """
        Sets values for five G2++ interest rate parameters: "a", "b", "sigma", "eta", "rho" and calculates five
        dependent parameters "miu_x", "miu_y", "sigma_x","sigma_y","rho_xy"  based interest rate paramteres.
        This function should be called prior to any pricing procedure with proper parameters.
        This function is also called (iteratively) in parameter-calibration process.

        Reference:

        Parameters:
        ------------------------------------
        G2_param: dictionary with keys:"a", "b", "sigma", "eta", "rho" and float values. e.g.
                   G2_param={
                   "a": 0.2
                   "b":0.1
                   "sigma":0.01
                   "eta":0.015
                   "rho":-0.75
                   }

        self.maturity: maturity of swaption, quoted in year; i.e. maturity=10 for swaption (10, tenor); float or int

        Returns:
        -----------------------------------
        0(Not being used), and self.param dictionary's values are updated.
        """
        # independent interest rate parameters
        self.param["a"] = G2_param["a"]
        self.param["b"] = G2_param["b"]
        self.param["sigma"] = G2_param["sigma"]
        self.param["eta"] = G2_param["eta"]
        self.param["rho"] = G2_param["rho"]

        # dependent  parameters generated with interest rate parameters
        # they HAVE TO be called in this order
        self.param["miu_x"] = -1 * self.M_x(0, self.maturity)
        self.param["miu_y"] = -1 * self.M_y(0, self.maturity)
        self.param["sigma_x"] = self.sigma_x()
        self.param["sigma_y"] = self.sigma_y()
        self.param["rho_xy"] = self.rho_xy()

        # print (self.param)
        return 0
        # set 5 interest rate prameters

    def calc_price_lognorm(self, sigma_ln):
        """
        :param sigma_ln:
        :param data:
        :return:
        """
        "level 0"
        from scipy.stats import norm

        forward_swap_rate, T0, Tn, disc_curve, freq = (self.strike, (self.maturity), (self.tenor), self.ini_curve, 12)
        B_vec = disc_curve
        K = forward_swap_rate  # at the money
        d1 = (np.log(forward_swap_rate / K) + 0.5 * sigma_ln ** 2 * T0) / (sigma_ln * np.sqrt(T0))
        d2 = (np.log(forward_swap_rate / K) - 0.5 * sigma_ln ** 2 * T0) / (sigma_ln * np.sqrt(T0))

        price_ln = (forward_swap_rate * norm.cdf(d1) - K * norm.cdf(d2)) * \
                   (np.sum([B_vec[int(ti * self.index_multipler)] for ti in
                            self.maturity + np.arange(1, self.tenor + 1)]))

        return price_ln * self.nominal

    def calc_price_lognorm_solve(self, sigma_ln, *data):
        """
        :param sigma_ln:
        :param data:
        :return:
        """
        "level 0"
        from scipy.stats import norm

        forward_swap_rate, T0, Tn, disc_curve, given_price, freq = data
        B_vec = disc_curve
        K = forward_swap_rate  # at the money
        d1 = (np.log(forward_swap_rate / K) + 0.5 * sigma_ln ** 2 * T0) / (sigma_ln * np.sqrt(T0))
        d2 = (np.log(forward_swap_rate / K) - 0.5 * sigma_ln ** 2 * T0) / (sigma_ln * np.sqrt(T0))
        #        price_ln = (forward_swap_rate * norm.cdf(d1) - K * norm.cdf(d2)) * (
        #                    np.sum(B_vec[int(T0 * freq + 1):int( (T0 + Tn) * freq + 1)]) / freq)

        price_ln = (forward_swap_rate * norm.cdf(d1) - K * norm.cdf(d2)) * \
                   (np.sum(
                       [B_vec[int(ti * self.index_multipler)] for ti in self.maturity + np.arange(1, self.tenor + 1)]))

        return price_ln * 100. - given_price

    def calc_implied_vol(self, price, freq=12):
        """

        :param price:
        :return:
        """
        "level 0"
        from scipy.optimize import fsolve
        data = (self.strike, (self.maturity), (self.tenor), self.ini_curve, price, freq)
        sigma_ln0 = 0.20
        signam_ln_solved = fsolve(self.calc_price_lognorm_solve, sigma_ln0, args=data)
        return signam_ln_solved[0]

    def swaption_price_(self):

        """
        Calculates integration value of computed integrand, and adjusted by 1. payer/receiver identifier 2. dual-curve adjustment
        3. nominal amount 4. discount factor B(0,maturity) for present value
        return: present value of a swaption
        """
        from scipy.integrate import quad, quadrature

        "level 1 "
        import time
        start_time = time.time()

        intergrand_lower_bd = self.param["miu_x"] - 10 * self.param["sigma_x"]
        intergrand_upper_bd = self.param["miu_x"] + 10 * self.param["sigma_x"]
        quad_num = 80

        quad_space = np.linspace(intergrand_lower_bd, intergrand_upper_bd, quad_num)
        delta_t = ((intergrand_upper_bd - intergrand_lower_bd) / quad_num)
        integration = sum(self.integrand(quad_space) * delta_t)

        swaption_price = self.omega * self.adj * self.nominal * self.ini_curve[
            int(self.index_multipler * self.maturity)] * integration
        # swaption_price=self.omega*self.adj*self.nominal*self.ini_curve[int(self.index_multipler*self.maturity)]*quad(self.integrand,a=self.param["miu_x"]-10*self.param["sigma_x"],b=self.param["miu_x"]+10*self.param["sigma_x"])[0]
        swaption_implied_vol = self.calc_implied_vol(swaption_price)
        time_cost = time.time() - start_time

        return {"val": swaption_price, "time_cost": time_cost, "vol": swaption_implied_vol}
        # return s

    def integrand(self, x):
        """

        :param x:
        :return:
        """
        "level 2"

        from scipy.stats import norm

        h1 = self.h1(x)

        integrand_part_1 = np.exp(-0.5 * ((x - self.param["miu_x"]) / self.param["sigma_x"]) ** 2) / (
                    self.param["sigma_x"] * np.sqrt(2 * np.pi))
        integrand_part_2 = norm.cdf(-1 * self.omega * h1) - np.sum(
            [self.lamda_i(x, t_i, c_i) * np.exp(self.k_i(x, t_i)) * norm.cdf(-1 * self.omega * self.h2_i(h1, t_i)) for
             t_i, c_i in zip(self.pmtdates, self.ci)], axis=0)

        return integrand_part_1 * integrand_part_2

    def h1(self, x):
        """

        :param x:
        :return:
        """
        "level 3"
        try:

            y_hat = list(map(self.solver, x))
            y_hat = np.array(y_hat)
        except TypeError:

            y_hat = self.solver(x)

        return (y_hat - self.param["miu_y"]) / (self.param["sigma_y"] * np.sqrt(1 - self.param["rho_xy"] ** 2)) - \
               self.param["rho_xy"] * (x - self.param["miu_x"]) / (
                           self.param["sigma_x"] * np.sqrt(1 - self.param["rho_xy"] ** 2))

    # h1(x)=some function of above vars

    def h2_i(self, h1, t_i):
        """

        :param h1:
        :param t_i:
        :return:
        """

        "level 3"

        h1 = h1
        B = self.B_b(self.maturity, t_i)
        return h1 + B * self.param["sigma_y"] * np.sqrt(1 - self.param["rho_xy"] ** 2)

    def lamda_i(self, x, t_i, c_i):
        """

        :param x:
        :param t_i:
        :param c_i:
        :return:
        """
        "level 4"

        A = self.A(self.maturity, t_i)
        B = self.B_a(self.maturity, t_i)

        return c_i * A * np.exp(-B * x)

    def k_i(self, x, t_i):
        """

        :param x:
        :param t_i:
        :return:
        """

        "level 4"

        B = self.B_b(self.maturity, t_i)
        return -B * (self.param["miu_y"] - 0.5 * (1 - self.param["rho_xy"] ** 2) * self.param["sigma_y"] ** 2 * B
                     + self.param["rho_xy"] * self.param["sigma_y"] * (x - self.param["miu_x"]) / self.param["sigma_x"])

    def solver(self, x):
        """

        :param x:
        :return:
        """
        "level 4"
        from scipy.optimize import newton

        y_hat = newton(self.equal, args=(x,), x0=0.02, maxiter=100)

        return y_hat

    def equal(self, y_hat, x):
        """

        :param y_hat:
        :param x:
        :return:
        """
        "level 5"

        equation = np.sum([c_i * self.A(self.maturity, t_i) * np.exp(
            -self.B_a(self.maturity, t_i) * x - self.B_b(self.maturity, t_i) * y_hat) for
                           t_i, c_i in zip(self.pmtdates, self.ci)]) - 1
        return equation

    def A(self, T, t_i):
        """

        :param T:
        :param t_i:
        :return:
        """
        "level 4"

        return (self.ini_curve[int(self.index_multipler * t_i)] / self.ini_curve[int(self.index_multipler * T)]) * \
               np.exp(0.5 * (self.V(T, t_i) - self.V(0, t_i) + self.V(0, T)))

    def B_a(self, T, t_i):
        """
        :param T:
        :param t_i:
        :return:
        """
        "level 4"

        return (1 - np.exp(-self.param["a"] * (t_i - T))) / self.param["a"]

    def B_b(self, T, t_i):
        """

        :param T:
        :param t_i:
        :return:
        """
        "level 4"

        return (1 - np.exp(-self.param["b"] * (t_i - T))) / self.param["b"]

    def V(self, T, t_i):
        """

        :param T:
        :param t_i:
        :return:
        """
        "level 5"

        s = t_i - T
        a = self.param["a"]
        b = self.param["b"]
        sigma = self.param["sigma"]
        eta = self.param["eta"]
        rho = self.param["rho"]
        function = (sigma ** 2 / a ** 2) * (
                s + 2. * np.exp(-a * s) / a - np.exp(-2. * a * s) / (2 * a) - 3. / (2 * a)) \
                   + (eta ** 2 / b ** 2) * (
                           s + 2. * np.exp(-b * s) / b - np.exp(-2. * b * s) / (2 * b) - 3. / (2 * b)) \
                   + (2. * rho * sigma * eta / (a * b)) * (
                           s + (np.exp(-a * s) - 1) / a + (np.exp(-b * s) - 1)
                           / b - (np.exp(-(a + b) * s) - 1) / (b + a))
        return function

    def M_x(self, s, t):
        """

        Calculates dependent variable "M_x" based on G2++ interest rate parameters(see helper of self.set_parameter ).
        param s: ALWAYS fixed at 0
        param t: Always fixed at self.maturity
        return:m_x float
        """
        "level 0"

        a = self.param["a"]
        b = self.param["b"]
        sigma = self.param["sigma"]
        eta = self.param["eta"]
        rho = self.param["rho"]
        m_x = (sigma ** 2 / a ** 2 + rho * sigma * eta / a / b) * (
                    1 - np.exp(-a * (t - s))) - 0.5 * sigma ** 2 / a ** 2 * (
                          np.exp(- a * (self.maturity - t)) - np.exp(- a * (self.maturity + t - 2 * s))) \
              - rho * sigma * eta / (b * (a + b)) * (
                          np.exp(-b * (self.maturity - t)) - np.exp(-b * self.maturity - a * t + (a + b) * s))
        return m_x

    def M_y(self, s, t):
        """
        Calculates dependent variable "M_y" based on G2++ interest rate parameters(see helper of self.set_parameter ).
        param s: ALWAYS fixed at 0
        param t: Always fixed at self.maturity
        return: m_y float
        """
        "level 0"

        a = self.param["a"]
        b = self.param["b"]
        sigma = self.param["sigma"]
        eta = self.param["eta"]
        rho = self.param["rho"]
        m_y = (eta ** 2 / b ** 2 + rho * sigma * eta / a / b) * (1 - np.exp(-b * (t - s))) \
              - 0.5 * eta ** 2 / b ** 2 * (np.exp(-b * (self.maturity - t)) - np.exp(-b * (self.maturity + t - 2 * s))) \
              - rho * sigma * eta / (a * (a + b)) * (
                          np.exp(-a * (self.maturity - t)) - np.exp(-a * self.maturity - b * t + (a + b) * s))

        return m_y

    def sigma_x(self):
        """
        Calculates dependent variable "sigma_x" based on G2++ interest rate parameters(see helper of self.set_parameter ).
        return: sigma_x float
        """
        "level 0"

        sigma_x = self.param["sigma"] * np.sqrt(
            (1 - np.exp(-2 * self.param["a"] * self.maturity)) / (2 * self.param["a"]))

        return sigma_x

    def sigma_y(self):
        """
        Calculates dependent variable "sigma_y" based on G2++ interest rate parameters(see helper of self.set_parameter ).
        return: sigma_y float
        """

        sigma_y = self.param["eta"] * np.sqrt(
            (1 - np.exp(-2 * self.param["b"] * self.maturity)) / (2 * self.param["b"]))
        return sigma_y

    def rho_xy(self):
        """
        Calculates dependent variable "rho_xy" based on G2++ interest rate parameters(see helper of self.set_parameter ).

        """
        "level 0"

        numerator = (self.param["rho"] * self.param["sigma"] * self.param["eta"]) * (
                    1 - np.exp(-1 * (self.param["a"] + self.param["b"]) * self.maturity))
        denominator = (self.param["a"] + self.param["b"]) * self.param["sigma_x"] * self.param["sigma_y"]
        rho_xy = numerator / denominator
        return rho_xy
