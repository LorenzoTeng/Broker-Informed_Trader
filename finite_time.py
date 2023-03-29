#This code is part of realization of paper
#Brokers and Informed Traders: dealing with toxic flow and extracting trading signal
#by A ́lvaro Cartea, and Leandro S ́anchez-Betancourt
#code by houhanteng s2447087

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import quad


class Finite_Model:

    def __init__(self, midprice_para, alpha_para, uninf_para, brok_pref, inf_pref, spread, time = 1, steps = 10000):
        #this model assuming initial value of non-constant process except midprice are 0
        self.capt = time
        self.steps = steps
        self.dt = time / steps
        self.timescale = math.sqrt(self.dt)

        self.initial = midprice_para[0]
        self.sigma_s = midprice_para[1]
        self.b = midprice_para[2]

        self.kappa_a = alpha_para[0]
        self.sigma_a = alpha_para[1]

        self.kappa_u = uninf_para[0]
        self.sigma_u = uninf_para[1]

        self.kb = spread[0]
        self.ki = spread[1]
        self.ku = spread[2]

        self.ab = brok_pref[0]
        self.phib = brok_pref[1]

        self.ai = inf_pref[0]
        self.phii = inf_pref[1]
        self.pso = inf_pref[2]#ambuiguity aversion

        #calculation of time coeffecient functions irrelevant to scenarios
        self.cap_phi = self.pso + (0.5 * self.phii* self.sigma_s * self.sigma_s)
        self.gamma1 = math.sqrt(self.cap_phi / self.ki)
        self.zeta1 = (self.ai
                      + math.sqrt(self.ki * self.cap_phi))\
                     / \
                     (self.ai
                      - math.sqrt(self.ki * self.cap_phi))

        self.gamma2 = math.sqrt(self.phib / self.kb)
        self.zeta2 = (self.ab
                      - self.b/2
                      + math.sqrt(self.phib * self.kb))\
                     / \
                     (self.ab
                      - self.b/2
                      - math.sqrt(self.phib * self.kb))

        self.g0 = np.array([self.g0t(i * self.dt) for i in range(self.steps)])
        self.g1 = np.array([self.g1t(i * self.dt) for i in range(self.steps)])
        self.q2 = np.array([self.q2t(i * self.dt) for i in range(self.steps)])
        self.p2 = np.array([self.p2t(i * self.dt) for i in range(self.steps)])
        self.p3 = np.array([self.p3t(i * self.dt) for i in range(self.steps)])
        self.p1 = self.p1_sequence()

    #time function
    def g0t(self, t):

        y = (
                (self.zeta1 * (math.exp(-self.kappa_a * (self.capt - t)) - math.exp(self.gamma1 * (self.capt - t)))
                 / (self.kappa_a + self.gamma1))
            -
                ((math.exp(-self.kappa_a * (self.capt - t)) - math.exp(-self.gamma1 * (self.capt - t)))
                 / (self.kappa_a - self.gamma1))

            ) \
            / ((math.exp(-self.gamma1 * (self.capt - t))
                - self.zeta1 * math.exp(self.gamma1 * (self.capt - t)))
                    * (2 * self.ki))
        return y

    def g1t(self, t):
        y = self.gamma1*(
                            (self.zeta1*math.exp(self.gamma1*(self.capt-t)) + math.exp(-self.gamma1*(self.capt - t))
                            )/(self.zeta1*math.exp(self.gamma1*(self.capt-t)) - math.exp(-self.gamma1*(self.capt - t)))
                        )
        return y

    def q2t(self, t):
        y = (math.sqrt(self.kb * self.phib) * (
                                                (1 + self.zeta2*math.exp(2*self.gamma2*(self.capt-t))
                                                 )/(
                                                    1 - self.zeta2*math.exp(2*self.gamma2*(self.capt-t)))
                                                )) \
             - self.b/2
        return y

    def p2t(self, t):
        coef = -2*self.gamma1/((self.zeta1 - math.exp(-2*self.gamma1*(self.capt - t)))
                               *(self.zeta2 - math.exp(-2*self.gamma2*(self.capt - t))))
        term1 = ((math.sqrt(self.kb * self.phib) - (self.b/2))/(self.gamma1 + self.gamma2)
                 )*(math.exp(-(self.gamma1 + self.gamma2)*(self.capt - t))
                    - math.exp(-2*(self.gamma1 + self.gamma2)*(self.capt - t)))
        term2 = self.zeta2*((math.sqrt(self.kb * self.phib) + (self.b/2))/(self.gamma1 - self.gamma2)
                            )*(math.exp(-(self.gamma1 + self.gamma2)*(self.capt - t))
                               - math.exp(-2*self.gamma1*(self.capt - t)))
        term3 = self.zeta1*((math.sqrt(self.kb * self.phib) - (self.b/2))/(self.gamma2 - self.gamma1)
                            )*(math.exp(-(self.gamma1 + self.gamma2)*(self.capt - t))
                               - math.exp(-2*self.gamma2*(self.capt - t)))
        term4 = self.zeta1*self.zeta2*((math.sqrt(self.kb * self.phib) + (self.b/2))/(self.gamma2 + self.gamma1)
                                       )*(math.exp(-(self.gamma1 + self.gamma2)*(self.capt - t))
                                          - 1)
        y = coef * (term1 + term2 + term3 - term4)
        return y

    def p3t(self, t):
        coef1 = (self.zeta2*(2*math.sqrt(self.kb * self.phib) + self.b))/(self.gamma2 + self.kappa_u)
        coef2 = (2*math.sqrt(self.kb * self.phib) - self.b)/(self.gamma2 - self.kappa_u)
        term1 = (1 - math.exp(-(self.kappa_u + self.gamma2)*(self.capt - t))
                 )/(self.zeta2 - math.exp(-2*self.gamma2*(self.capt - t)))
        term2 = (math.exp(-(self.kappa_u + self.gamma2)*(self.capt - t)) - math.exp(-2*self.gamma2*(self.capt - t))
                 )/(self.zeta2 - math.exp(-2*self.gamma2*(self.capt - t)))
        y = coef1 * term1 + coef2 * term2
        return y

    def p1_integrand(self, t, c):
        #Notice:t represent u variabl in paper formula, c represent t
        coef = math.exp(-self.kappa_a * (t - c))
        term1 = (math.exp(-self.gamma2*(self.capt-t))
                 - self.zeta2*math.exp(self.gamma2*(self.capt-t)))\
                /(math.exp(-self.gamma2*(self.capt-c))
                 - self.zeta2*math.exp(self.gamma2*(self.capt-c)))
        term2 = (self.g0t(t)*self.p2t(t) -2*self.g0t(t)*self.q2t(t) + 1)
        y = coef*term1*term2
        return y

    def p1_sequence(self):
        #produce a sequence
        p1 = []
        for i in range(self.steps):
            t = i * self.dt
            c = i * self.dt
            integral = quad(self.p1_integrand, t, self.capt, args=(c))[0]
            p1.append(integral)
        p1 = np.array(p1)
        return p1


    #strategy & flow
    def simu_ou(self, type = 0):
        #zero mean orstein uhlenbeck
        process = [0]
        if type == 0:#case of alpha
            for i in range(self.steps - 1):
                increments = -self.kappa_a * process[i] * self.dt + self.sigma_a * self.timescale * np.random.normal(0, 1)
                value = process[i] + increments
                process.append(value)
        else:#case of uninformed
            for i in range(self.steps - 1):
                increments = -self.kappa_u * process[i] * self.dt + self.sigma_u * self.timescale * np.random.normal(0, 1)
                value = process[i] + increments
                process.append(value)
        return process

    def simu_midprice(self, alpha, brok_stra = []):
        process = [self.initial]
        if len(brok_stra) == 0:
            for i in range(self.steps - 1):
                increments = alpha[i] * self.dt + self.sigma_s * self.timescale * np.random.normal(0, 1)
                value = process[i] + increments
                process.append(value)
        else:
            for i in range(self.steps - 1):
                drift = alpha[i] - self.b * brok_stra[i]
                increments = drift * self.dt + self.sigma_s * self.timescale * np.random.normal(0, 1)
                value = process[i] + increments
                process.append(value)
        return process

    def fini_stra_inf(self, alpha, q, t):
        #t: an int in range(steps)
        #alpha,q: inventory signal at time t
        v = self.g0t(t) * alpha - self.g1t(t) * q
        return v

    def opt_inf(self, alpha):
        #alpha: process
        inventory = [0]
        strategy = []
        for i in range(self.steps):
            v = self.fini_stra_inf(alpha[i], inventory[i], i*self.dt)
            q = inventory[i] + v * self.dt
            strategy.append(v)
            inventory.append(q)
            #strategy is one time step ahead
        return inventory, strategy

    def opt_brok_vi(self, inve_i, stra_i, stra_u):
        inve_b = [0]
        stra_b = []
        for i in range(self.steps):
            term_vi = (self.p1[i]/(2*self.kb*self.g0[i]))*stra_i[i]
            term_qb = ((self.b + 2*self.q2[i])/(2*self.kb))*inve_b[i]
            term_qi = ((self.p2[i]+ ((self.p1[i]*self.g1[i])/self.g0[i]))/(2*self.kb))*inve_i[i]
            term_vu = (self.p3[i]/(2*self.kb)) * stra_u[i]
            stra_b_t = [term_vi, term_qb, term_qi, term_vu]
            inve_b_t = sum(stra_b_t) * self.dt + inve_b[i]
            inve_b.append(inve_b_t)
            stra_b.append(stra_b_t)
        return inve_b, stra_b

    def opt_brok_al(self, inve_i, alpha, stra_u):
        inve_b = [0]
        stra_b_bar = []
        for i in range(self.steps):
            term_al = (self.p1[i]/(2*self.kb))*alpha[i]
            term_qb = ((self.b + 2*self.q2[i])/(2*self.kb))*inve_b[i]
            term_qi = ((self.p2[i])/(2*self.kb)) * inve_i[i]
            term_vu = (self.p3[i]/(2*self.kb)) * stra_u[i]
            stra_b_t_bar = [term_al, term_qb, term_qi, term_vu]
            inve_b_t = sum(stra_b_t_bar)*self.dt + inve_b[i]
            inve_b.append(inve_b_t)
            stra_b_bar.append(stra_b_t_bar)

        return inve_b, stra_b_bar

    def cal_cashflow(self, strategy, midprice, who):
        if who == "i":
            k = self.ki
        elif who == "u":
            k = self.ku
        else:
            strategy = [sum(strategy[i]) for i in range(self.steps)]
            k = self.kb
        drift = -(np.array(midprice) + k * np.array(strategy)) * np.array(strategy) * self.dt
        cash = [0]
        for i in range(self.steps):
            cash_t =  sum(drift[0:i+1])
            cash.append(cash_t)
        cash = np.array(cash)
        return cash

    def cal_cash_brok(self, stra_b, stra_i, stra_u, midprice):
        cash_informed = self.cal_cashflow(stra_i, midprice, "i")
        cash_uninformed = self.cal_cashflow(stra_u, midprice, "u")
        cash_broker = self.cal_cashflow(stra_b, midprice, "b")
        cash = - cash_informed - cash_uninformed + cash_broker
        return cash


def average_pl(n, model):
    #profit & loss
    pl_informed = []
    pl_uninformed= []
    pl_litmarket = []
    pl_brok = []
    for i in range(n):
        # generate senario
        scenario_alpha0 = model.simu_ou()
        scenario_uninformed = model.simu_ou(1)
        scenario_midprice = model.simu_midprice(scenario_alpha0)

        # generate strategy
        inven_i0, strat_i0 = model.opt_inf(scenario_alpha0)
        inven_n0, strat_b0 = model.opt_brok_al(inven_i0, scenario_alpha0, scenario_uninformed)

        #cal P&L
        pl_informed_i = -model.cal_cashflow(strat_i0, scenario_midprice, "i")[-1]
        pl_uninformed_i = -model.cal_cashflow(scenario_uninformed, scenario_midprice, "u")[-1]
        pl_litmarket_i = model.cal_cashflow(strat_b0, scenario_midprice, "b")[-1]
        pl_brok_i = pl_informed_i + pl_uninformed_i + pl_litmarket_i

        pl_informed.append(pl_informed_i)
        pl_uninformed.append(pl_uninformed_i)
        pl_litmarket.append(pl_litmarket_i)
        pl_brok.append(pl_brok_i)

    return np.array(pl_informed), np.array(pl_uninformed), np.array(pl_litmarket), np.array(pl_brok)





#Model parameter same as paper
midprice_p = [100, 1, 0.001]
alpha_p = [5, 1]
uninf_p = [15, 100]
brok_pref = [1, 0.01]
inf_pref = [1, 0.01, 0.01]
spread = [0.0012, 0.001, 0.001]

#establish model
model1 = Finite_Model(midprice_p, alpha_p, uninf_p, brok_pref, inf_pref, spread)

#generate one scenario
scenario_alpha = model1.simu_ou()
scenario1_uninformed = model1.simu_ou(1)
scenario1_midprice = model1.simu_midprice(scenario_alpha)

inven_i, strat_i = model1.opt_inf(scenario_alpha)
inven_b_al, strat_b_al = model1.opt_brok_al(inven_i, scenario_alpha, scenario1_uninformed)
inven_b_vi, strat_b_vi = model1.opt_brok_vi(inven_i, strat_i, scenario1_uninformed)

cash_i = model1.cal_cashflow(strat_i, scenario1_midprice, "i")
cash_u = model1.cal_cashflow(scenario1_uninformed, scenario1_midprice, "u")
cash_b_vi = model1.cal_cash_brok(strat_b_vi, strat_i, scenario1_uninformed, scenario1_midprice)
cash_b_al = model1.cal_cash_brok(strat_b_al, strat_i, scenario1_uninformed, scenario1_midprice)

# #plot figure1
# alpha = scenario_alpha
# qi = inven_i
# qb = inven_b_al
# vi = strat_i
# vu = scenario1_uninformed
# vb = np.sum(np.array(strat_b_al), axis=1)
# t = np.linspace(0, 0.9999, 10000)
# t1 = np.linspace(0, 1, 10001)
#
# label_variable=[r'$\alpha_{t}$',
#                 r'$q_{t}^{I}$',
#                 r'$q_{t}^{B}$',
#                 r'$v_{t}^{U}$',
#                 r'$v_{t}^{I}$',
#                 r'$v_{t}^{B}$']
# color_list=['skyblue', 'springgreen', 'springgreen', 'khaki', 'coral', 'limegreen']
# ylim_list= [[], [-3,3], [-3,3], [-75,75], [-75,75], [-75,75]]
# x = [t, t1, t1, t, t, t, t]
# y = [alpha, qi, qb, vu, vi, vb]
#
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.plot(x[i], y[i], color=color_list[i], label=label_variable[i])
#     plt.hlines(0, -0.1, 1.1, color='pink', linestyle="dashed")
#     if i >= 1:
#         plt.ylim(ylim_list[i])
#     plt.xlim(-0.1, 1.1)
#     plt.legend()
#
# plt.show()

# #plot figure2
# b_al = np.array(strat_b_al) #strategy of b with componets alpha
# b_vi = np.array(strat_b_vi) #strategy of b with componets vi
# label_variable=[r'$\tilde{\mathfrak{r}_0}(t) \alpha_t$',
#                 r'$-\tilde{\mathfrak{r}_1}(t) Q_t^{B *}$',
#                 r'$-\tilde{\mathfrak{r}_2}(t) Q_t^{I *}$',
#                 r'$\tilde{\mathfrak{r}_3}(t) \nu_t^U$',
#                 r'$\mathfrak{r}_0(t) \nu_t^{I *}$',
#                 r'$-\mathfrak{r}_1(t) Q_t^{B *}$',
#                 r'$-\mathfrak{r}_2(t) Q_t^{I *}$',
#                 r'$\mathfrak{r}_3(t) \nu_t^U$']
# color_list=['skyblue', 'coral', 'khaki', 'limegreen', 'skyblue', 'coral', 'khaki', 'limegreen']
#
# y = [b_al[:, 0], b_al[:, 1], b_al[:, 2], b_al[:, 3], b_vi[:, 0], b_vi[:, 1], b_vi[:, 2], b_vi[:, 3]]
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.plot(t, y[i], color=color_list[i], label=label_variable[i])
#     plt.hlines(0, -0.1, 1.1, color='pink', linestyle="dashed")
#     plt.xlim(-0.1, 1.1)
#     plt.ylim(-50, 50)
#     plt.legend()
#
# plt.show()




#establish model
model1 = Finite_Model(midprice_p, alpha_p, uninf_p, brok_pref, inf_pref, spread)

#plot figure 3
PL_inf, PL_uninf, PL_lit, PL_brok = average_pl(10000, model1)
PL = [PL_inf, PL_uninf, PL_lit, PL_brok]
color_list = ['skyblue', 'coral', 'khaki', 'limegreen']
title_list = ['Informed', 'Uninformed', 'Litmarket', 'Broker']

for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.hist(PL[i], range=(-10, 10), color=color_list[i])
    plt.axvline(PL[i].mean(), color=blue, linestyle='dashed', linewidth=1)
    plt.axvline(0, color='pink', linestyle='dashed', linewidth=1)
    plt.yticks([])
    plt.title(title_list[i])

plt.show()






