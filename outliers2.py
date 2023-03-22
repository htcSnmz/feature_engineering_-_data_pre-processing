# Aykırı Değerleri Yakalama
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset("diamonds")
df = df.select_dtypes(["int64", "float64"])
df = df.dropna()
df.head()

df_table = df["table"]
df_table.head()
sns.boxplot(x=df_table)
plt.show()

q1 = df_table.quantile(0.25)
q3 = df_table.quantile(0.75)
iqr = q3 - q1
alt_sinir = q1 - 1.5 * iqr
ust_sinir = q3 + 1.5 *  iqr

df_table[(df_table < alt_sinir) | (df_table > ust_sinir)]

# Aykırı Değer Problemini Çözmek
# Silme
df_table = pd.DataFrame(df_table)
df_table.shape
t_df = df_table[~((df_table < alt_sinir) | (df_table > ust_sinir)).any(axis=1)]
t_df.shape

# Ortalama ile Doldurma
df_table = df["table"]
#df_table = pd.DataFrame(df_table)
aykiri = ((df_table < alt_sinir) | (df_table > ust_sinir))
df_table[aykiri] = df_table.mean()

# Baskılama Yöntemi
df_table = df["table"]
df_table[aykiri]
alt_sinir
ust_sinir
df_table[(df_table < alt_sinir)] = alt_sinir
df_table[df_table > ust_sinir] = ust_sinir

# Çok Değişkenli Aykırı Gözlem Analizi (LOC)
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:10]
np.sort(df_scores)[0:10]
esik_deger = np.sort(df_scores)[13] # eşik değer kabul edildi
aykiri_tf = df_scores > esik_deger
yeni_df = df[df_scores > esik_deger] # aykırı olmayan değerler

# Baskılama Yöntemi
baski_degeri = df[df_scores == esik_deger]
aykirilar = df[df_scores < esik_deger]
res = aykirilar.to_records(index=False)
res[:] = baski_degeri.to_records(index=False)
aykirilar = pd.DataFrame(res, index=aykirilar.index)