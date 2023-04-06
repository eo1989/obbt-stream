# %% [python]
import numpy as np
import warnings
import matplotlib as mp
import matplotlib.pyplot as plt

# import math as mt
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, embed, interact_manual
import ipywidgets as widgets
from functools import reduce
warnings.filterwarnings('ignore')

# %% [markdown] {"incorrectly_encoded_metadata": "{markdown}"}
# ##### The 'classical' Central Limit Theorem states that if have IID variables where the variance of the marginal distribution is finite, then the sum of the variables is converging in distribution to a **Normal distribution**.
# Thus, if $X_1,X_2,\dots,X_n$ are IID variables where $\mathbb{E}(X_i)=\mu$ and $\mathbb{V}(X_i)=\sigma^2$, then for $W_n=\sum_{i=1}^n X_i$ we can write
# \begin{equation}
# W_n \overset{\rm d}{\longrightarrow} N(n\mu,\sqrt{n}\sigma), \nonumber
# \end{equation}
# where we used the fact that both the sum and the variance can be simply summed for independent variables. As we can see, when $n\rightarrow\infty$, we obtain a Normal distribution with infinitely large mean and variance.
# The usual trick is to consider a **standardised** version of the sum of the variables, where we shift the variable such that the mean becomes 0 and rescale the variable such that the variance becomes 1. In this case this can be done simply by defining $Z_n$ as
# \begin{equation}
# Z_n =\frac{\left(\sum_{i=1}^n X_i\right) -n\mu}{\sqrt{n}\sigma}=
# \frac{\frac{1}{n}\left(\sum_{i=1}^n X_i\right) - \mu}{\sigma/\sqrt{n}} =\frac{\sqrt{n}\left(\overline{X_n}-\mu\right)}{\sigma}
# \end{equation}
# which is converging to standard Normal in distribution
# \begin{equation}
# Z_n\overset{\rm d}{\longrightarrow}N(0,1).
# \end{equation}
# %% [python]
num_pts = 20_000
n = 50

# summing uniformly distributed random variables

# function that can sum up to a specific number of variables
# and return the distribution of the result. Also included
# the *standardisation* in the process (subtract the mean and divide by the std)
distr_list = [np.random.uniform(low=0, high=1, size=num_pts) for i in range(0, n)]
# %%
def sum_dist(distr_list, sum_size):
    s_distr = [
        sum([distr_list[i][j] for i in range(0, sum_size)])
        for j in range(0, len(distr_list[0]))
    ]
    s_mu, s_var = np.mean(s_distr), np.var(s_distr, ddof=1.0)
    c_distr = [(x - s_mu) / np.sqrt(s_var) for x in s_distr]
    return c_distr


sum_distr_list = [sum_dist(distr_list, num_in_sum) for num_in_sum in range(1, n + 1)]

plt.clf()
for s_distr in sum_distr_list[:2]:
    hist, bins = np.histogram(s_distr, bins=100, normed=True) # type: ignore
    plt.plot(bins[:-1], hist, "o", alpha=0.2)
    plt.ylim(0, 1.0)
plt.show()
# %%
def plot_sum_dist(sum_distr_list, chosen_distr_id):
    hist, bins = np.histogram(
        sum_distr_list[chosen_distr_id - 1], bins=100, normed=True
    )
    plt.plot(bins[:-1], hist)
    x_list = np.arange(-4, 4, 0.1)
    y_list = [np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi) for x in x_list]
    plt.plot(x_list, y_list)
    plt.xlabel(r"$\frac{x-\overline{X}}{\rm sd}$", fontsize = 10, loc='left')
    plt.ylim(0, 1.0)
    plt.xlim(-4, 4)
    plt.show()


# sum_distr_list = [sum_dist(distr_list, num_in_sum) for num_in_sum in range(1, n + 1)]
# TODO - Fix this ipywidget ValueError during runtime. : cannot find widget or abbreviation for argument: 'sum_distr_list'

# x = widgets.interact(
#     plot_sum_dist,
#     s_distr_list=widgets.fixed(sum_distr_list),
#     chosen_distr_id=widgets.IntSlider(
#         min=1, max=n, step=1, value=1, description="num. distr. summed"
#     ),
# )
# display(x)


interact(plot_sum_dist,
         s_distr_list=fixed(sum_distr_list),
         chosen_distr_id=IntSlider(
             min=1,
             max=n,
             step=1,
             value=1,
            description="num. distr. summed"
            )
        )


# %%

exp_distr_list = [np.random.exponential(3.0, num_pts) for i in range(0, n)]
sum_exp_distr = [sum_dist(exp_distr_list, num_in_sum) for num_in_sum in range(1, n + 1)]

widgets.interact(
    plot_sum_dist,
    s_distr_list=widgets.fixed(sum_distr_list),
    chosen_distr_id=widgets.IntSlider(
        min=1, max=n, step=1, value=1, description="num. distr. summed"
    ),
)

# %%
# Summing heavy tailed distributions

pareto_a = 1.5
pareto_dist_list = [np.random.pareto(pareto_a, num_pts) for i in range(0, n)]

# %% [markdown]
"""
Key difference is that if the PDF of the marginal is decaying as $x^{\neg\alpha}$ (where $2<\alpha<3$), then we have to 'standardise' the sum by dividing with $n^{\frac{1}{\mu}}$ where $\mu={\alpha-1}$.
Thus the summing function in this case has to take an argument corresponding to \mu as well.
"""
# %%

def sum_pareto(distr_list, sum_size, mu):
    s_distr = [
        sum([distr_list[i][j]] for i in range(0, sum_size)) # type: ignore
        for j in range(0, len(distr_list[0]))
    ]
    s_mu = np.mean(s_distr)
    c_distr = [(x - s_mu) / sum_size ** (1.0 / mu) for x in s_distr]
    return c_distr


# %%

# mu = 1.0
# sum_par_distr = [sum_pareto(pareto_dist_list, n, mu) for n in range(1, n + 1)]
# plt.clf()
# for s_distr in sum_par_distr:
#     hist, bins = np.histogram(s_distr, bins = np.logspace(0, 3, 50), density=True)
#     plt.loglog(bins[:-1], hist, 'o', alpha = 0.2)
#     plt.title(f"$\alpha$ = {pareto_a + 1}, $\mu$ = {mu}")
# plt.show()


mu = 1.0
sum_par_distr = [sum_pareto(pareto_dist_list, n, mu) for n in range(1, n + 1)]
plt.clf()
for s_distr in sum_par_distr:
    hist, bins = np.histogram(s_distr, bins=np.logspace(0, 3, 50), density=True)
    plt.loglog(bins[:-1], hist, "o", alpha=0.2)
    plt.title(r"$\alpha$=" + str(pareto_a + 1) + ", $\mu$=" + str(mu))
plt.show()

mu = 1.5
sum_par_distr = [sum_pareto(pareto_dist_list, n, mu) for n in range(1, n + 1)]
plt.clf()
for s_distr in sum_par_distr:
    hist, bins = np.histogram(s_distr, bins=np.logspace(0, 3, 50), density=True)
    plt.loglog(bins[:-1], hist, "o", alpha=0.2)
    plt.title(r"$\alpha$=" + str(pareto_a + 1) + ", $\mu$=" + str(mu))
plt.show()

mu = 2.0
sum_par_distr = [sum_pareto(pareto_dist_list, n, mu) for n in range(1, n + 1)]
plt.clf()
for s_distr in sum_par_distr:
    hist, bins = np.histogram(s_distr, bins=np.logspace(0, 3, 50), density=True)
    plt.loglog(bins[:-1], hist, "o", alpha=0.2)
    plt.title(r"$\alpha$=" + str(pareto_a + 1) + ", $\mu$=" + str(mu))
plt.show()

mu = 2.5
sum_par_distr = [sum_pareto(pareto_dist_list, n, mu) for n in range(1, n + 1)]
plt.clf()
for s_distr in sum_par_distr:
    hist, bins = np.histogram(s_distr, bins=np.logspace(0, 3, 50), density=True)
    plt.loglog(bins[:-1], hist, "o", alpha=0.2)
    plt.title(r"$\alpha$=" + str(pareto_a + 1) + ", $\mu$=" + str(mu))
plt.show()
