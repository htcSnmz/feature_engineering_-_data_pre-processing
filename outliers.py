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
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("../datasets/titanic.csv")
df.head()

# 1) Aykırı Değerler
# Aykırı Değerleri Yakalama
# Grafik Teknikle Aykırı Değerler
sns.boxplot(df["Age"])
plt.show()

# Aykırı Değerler Nasıl Yakalanır?
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr
df[(df["Age"] > up) | (df["Age"] < low)]
df[(df["Age"] > up) | (df["Age"] < low)].index

# Aykırı Değer Var mı Yok mu ?
df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)

# İşlemleri Fonksiyonlaştırmak
def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

low, up = outlier_threshold(df, "Age")
df[(df["Age"] > up) | (df["Age"] < low)].head()
# low, up = outlier_threshold(df, "Fare")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Fare")
check_outlier(df, "Age")

# check_outlier fonksiyonunu verimli kullanabilmek için sayısal değişkenleri belirleme (grab_col_names)
def grab_col_names(dataframe, cat_th=10, car_th=20):
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
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
for col in num_cols:
    print(col + ":", check_outlier(df, col))

dff = pd.read_csv("../datasets/application_train.csv")
cat_cols, num_cols, cat_but_car = grab_col_names(dff)
for col in num_cols:
    print(col + ":", check_outlier(dff, col))

# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_threshold(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")
age_index = grab_outliers(df, "Age", True)

# Aykırı Değer Problemini Çözme
# Silme
low, up = outlier_threshold(df, "Fare")
df.shape
df[~((df["Fare"] > up) | (df["Fare"] < low))]
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape
for col in num_cols:
   new_df =  remove_outlier(df, col)

new_df.shape
df.shape[0] - new_df.shape[0] # 116 gözlem silimmiş

# Baskılama Yöntemi (re-assignment with thresholds)
low, up = outlier_threshold(df, "Fare")
#df.loc[df["Fare"] > up, "Fare"] = up
#df.loc[df["Fare"] < low, "Fare"] = low
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit


# Recap
df = pd.read_csv("../datasets/titanic.csv")
outlier_threshold(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)
remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")

# Çok Değişkenli Aykırı Değer Analizi (Local Outlier Factor (LOF))
df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()
for col in df.columns:
    print(col + ":", check_outlier(df, col))
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5] # -1'den uzaklaştıkça outlier olma durumu artar.
# Eşik değer belirleme için elbow yöntemi
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,20], style=".-") #3. indeksteki değer eşik değer
plt.show()
threshold = np.sort(df_scores)[3]
df[df_scores < threshold]
df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T
df[df_scores < threshold].drop(axis=0, labels=df[df_scores < threshold].index)