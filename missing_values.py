# !pip install missingno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = sns.load_dataset("titanic")
df.head()

# Eksik Değerlerin Yakalanması
# eksik gözlem var mı yok mu sorgusu
df.isnull().values.any()
# değişkenlerdeki eksik değer sayısı
df.isnull().sum()
# değişkenlerdeki tam değer sayısı
df.notnull().sum()
# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()
# en az bir tane eksik değer sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]
# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]
# değişkenlerin yüzdesel olarak eksik değer oranı
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
# eksik değere sahip değişken isimlerinin yakalanması
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
# fonksiyonlaştırma
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

str(df["embarked"].dtype)
# Eksik Değer Problemini Çözme
# hızlıca silmek
df.dropna()
# basit atama yöntemleri ile doldurmak
df["age"].fillna(df["age"].mean())
df["age"].fillna(df["age"].median())
# sayısal değişkenleri ortalama ile doldurmak
dff = df.apply(lambda x: x.fillna(x.mean()) if str(x.dtype) not in ["object", "category"] else x, axis=0)
dff.isnull().sum()
# kategorik değişkenler için mod ile doldurmak
df["embarked"].fillna(df["embarked"].mode()[0])
df["embarked"].fillna(df["embarked"].mode()[0]).isnull().sum()
dff = dff.apply(lambda x: x.fillna(x.mode()[0]) if (str(x.dtype) in ["object", "category"] and x.nunique() <= 10) else x, axis=0)


# Kategorik Değişken Kırılımında Değer Atama
df["age"].fillna(df.groupby("sex")["age"].transform("mean"))
df["age"].fillna(df.groupby("sex")["age"].transform("mean")).isnull().sum()
# 2. yol
# df.loc[(df["age"].isnull()) & (df["sex"] == "female"), "age"] = df.groupby("sex")["age"].mean()["female"]


# Tahmine Dayalı Atama İşlemi
def grab_col_names(dataframe, cat_th=10, car_th=30):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"] and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_bat_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
# knn'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
df["age_imputed_knn"] = dff[["age"]]
df.loc[df["age"].isnull(), ["age", "age_imputed_knn"]]


# Eksik Veri Yapısını İncelemek
msno.bar(df)
plt.show()
msno.matrix(df)
plt.show()
msno.heatmap(df)
plt.show()


# Eksik Değerlerin Bağımlı Değişken ile Analizi
na_cols = missing_values_table(df, True)
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "COUNT": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "survived", na_cols)