# Label Encoding & Binary Encoding
##########################################
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset("titanic")
df.head()
df["sex"].head()
le = LabelEncoder()
le.fit_transform(df["sex"])[0:5] # alfabetik sıraya göre kodlanır. (0: female, 1:male)
# Hangi koda hangi değer karşılık geliyor?
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()


# One Hot Encoding
##########################################
df["embarked"].value_counts()
pd.get_dummies(df, columns=["embarked"], drop_first=True).head()
# dummy_na = True yaparsak eksik değerler için de dummy değ. oluşur.

pd.get_dummies(df, columns=["sex"], drop_first=True).head() # binary code olur.

pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()


# Rare Encoding
##########################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
# 3. Rare encoder yazılması.

# 1
df.embark_town.value_counts()

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
cat_cols

def cat_summary(dataframe, col_name, plot=False):
    df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    print(df)
    print("#######################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

# 2
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "survived", cat_cols)

# 3
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "survived", cat_cols)


# Feature Scaling
##########################################

# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, std sapmaya böl. z = (x - u) / s
ss = StandardScaler()
df["age_std_scaler"] = ss.fit_transform(df[["age"]])
df.head()

# RobustScaler: Medyanı çıkar, iqr'a böl. Aykırı değerlerden daha az etkilenir.
rs = RobustScaler()
df["age_robust_scaler"] = rs.fit_transform(df[["age"]])
df.describe().T

# Min-Max Scaler: Verilen 2 değer arasında değişken dönüşümü
mms = MinMaxScaler()
df["age_min_max_scaler"] = mms.fit_transform(df[["age"]])
df.describe().T

# Sayısal Değişkenleri Kategorik Değişkenlere Çevirme (Binning)
df["age_qcut"] = pd.qcut(df["age"], 5)

