# %%
import matplotlib.pyplot as plt
import numpy as np

# from openbb_terminal.sdk import openbb as ob
from scipy.stats import norm, gamma
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# %%
# "S" => "sample"
# 20,000 samples per sample size
S_5 = np.zeros(20_000)
S_10 = np.zeros(20_000)
S_100 = np.zeros(20_000)
S_250 = np.zeros(20_000)

# usual avg, std of population distribution
mu, sigma = 15, 2

# %%
for A in range(20_000):
    S = np.random.normal(mu, sigma, 5)
    X_bar_5 = S.mean()
    S_5[A] = X_bar_5

for B in range(20_000):
    S = np.random.normal(mu, sigma, 10)
    X_bar_10 = S.mean()
    S_10[B] = X_bar_10

for C in range(20_000):
    S = np.random.normal(mu, sigma, 100)
    X_bar_100 = S.mean()
    S_100[C] = X_bar_100

for D in range(20_000):
    S = np.random.normal(mu, sigma, 250)
    X_bar_250 = S.mean()
    S_250[D] = X_bar_250

# %%
df = pd.DataFrame(np.vstack((S_5, S_10, S_100, S_250)))
df = df.T
df.columns = ["5", "10", "100", "250"]

plt.figure(figsize=(20, 10))
sns.distplot(df["5"])
sns.distplot(df["10"])
sns.distplot(df["100"])
sns.distplot(df["250"])
plt.title("Students per Classroom distribution", fontsize=20)
plt.xlabel('Sample Mean', fontsize = 20)
plt.ylabel('Density', fontsize = 20 )
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(df, fontsize = 20)
plt.show()
# %%

df_random = pd.read_csv('../../../../../../mnt/c/Users/eorlo/Desktop/spy_3.csv', sep=',')
# df_random

# for row in df_random.index:
#     print(row, end = ', ')

# list(df_random.columns)
# columns w/ a ".1" suffix are to represent SPY Put values for those expirations && strikes.
# reformat them into two separate dataframes, one for calls, one for puts.

# %%
