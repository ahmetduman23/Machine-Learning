###################################################
# PROJECT: SALARY PREDICTİON WITH MACHINE LEARNING
###################################################

# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör



############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################

df = pd.read_csv("7Week_ML_Part1/salary_prediction/Hitters.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    #print("##################### Tail #####################")
    #print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################

def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

df[num_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(df[num_cols].corr(), annot=True, linewidths=.5, ax=ax)
plt.show()


# correlation with the final state of the variables
plt.figure(figsize=(45,45))
corr=df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(df[num_cols].corr(), mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5,annot=True)
plt.show(block=True)

# kategorik degisken ordinal
# map - ortaokul -0 , lisans 1 , yükseklisans 2,


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "Salary":
            pass
        else:
            correlation = dataframe[[col, "Salary"]].corr().loc[col, "Salary"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)



#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)


#############################################
# 1. Outliers (Aykırı Değerler)
#############################################


sns.boxplot(x=df["Salary"], data=df)
plt.show(block=True)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


#############################################
# 2. Missing Values (Eksik Değerler)
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)


# Eksik veri analizine uygun olarak 3 farkli yöntem kullanabiliriz.
df1 = df.copy()
df1.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df1)


method = int(input("Eksik veri için hangi yöntemi uygulamak istersiniz? (1/2/3): "))

if method ==1:
    dff = pd.get_dummies(df1[cat_cols + num_cols], drop_first=True)
    scaler = RobustScaler()
    dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    imputer = KNNImputer(n_neighbors=5)
    dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
    dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
    df1 = dff


elif method ==2:
    df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df["Division"] == "E"), "Salary"] = \
    df1.groupby(["League", "Division"])["Salary"].mean()["A", "E"]

    df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df["Division"] == "W"), "Salary"] = \
    df1.groupby(["League", "Division"])["Salary"].mean()["A", "W"]

    df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df["Division"] == "E"), "Salary"] = \
    df1.groupby(["League", "Division"])["Salary"].mean()["N", "E"]

    df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df["Division"] == "W"), "Salary"] = \
    df1.groupby(["League", "Division"])["Salary"].mean()["N", "W"]


elif method == 3:
    # Drop NA
    # Eksik değer içeren tüm satırları silme
    df1.dropna(inplace=True)

print(df1.head())
print(df1.isnull().sum())

def eksik_veri_doldur(dataframe,method):
    df1 = dataframe.copy()
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    if method == 1:
        dff = pd.get_dummies(df1[cat_cols + num_cols], drop_first=True)
        scaler = RobustScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
        imputer = KNNImputer(n_neighbors=5)
        dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
        dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
        df1 = dff
        pass


    elif method == 2:
        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df["Division"] == "E"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["A", "E"]

        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "A") & (df["Division"] == "W"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["A", "W"]

        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df["Division"] == "E"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["N", "E"]

        df1.loc[(df1["Salary"].isnull()) & (df1["League"] == "N") & (df["Division"] == "W"), "Salary"] = \
            df1.groupby(["League", "Division"])["Salary"].mean()["N", "W"]
        pass

    elif method == 3:
        # Drop NA
        # Eksik değer içeren tüm satırları silme
        df1.dropna(inplace=True)
        pass
    return df1

df1 = eksik_veri_doldur(df,method=1)

#############################################
# 3. Feature Extraction (Özellik Çıkarımı)
#############################################

new_num_cols=[col for col in num_cols if col!="Salary"]
df1[new_num_cols]=df1[new_num_cols]+0.0000000001

df1["Hits_Success"] = (df1["Hits"] / df1["AtBat"]) * 100
df1['NEW_RBI'] = df1['RBI'] / df1['CRBI']
df1['NEW_Walks'] = df1['Walks'] / df1['CWalks']
df1['NEW_PutOuts'] = df1['PutOuts'] * df1['Years']
df1['NEW_Hits'] = df1['Hits'] / df1['CHits'] + df1['Hits']
df1["NEW_CRBI*CATBAT"] = df1['CRBI'] * df1['CAtBat']
df1["NEW_Chits"] = df1["CHits"] / df1["Years"]
df1["NEW_CHmRun"] = df1["CHmRun"] * df1["Years"]
df1["NEW_CRuns"] = df1["CRuns"] / df1["Years"]
df1["NEW_Chits"] = df1["CHits"] * df1["Years"]
df1["NEW_RW"] = df1["RBI"] * df1["Walks"]
df1["NEW_RBWALK"] = df1["RBI"] / df1["Walks"]
df1["NEW_CH_CB"] = df1["CHits"] / df1["CAtBat"]
df1["NEW_CHm_CAT"] = df1["CHmRun"] / df1["CAtBat"]
df1['NEW_Diff_Atbat'] = df1['AtBat'] - (df1['CAtBat'] / df1['Years'])
df1['NEW_Diff_Hits'] = df1['Hits'] - (df1['CHits'] / df1['Years'])
df1['NEW_Diff_HmRun'] = df1['HmRun'] - (df1['CHmRun'] / df1['Years'])
df1['NEW_Diff_Runs'] = df1['Runs'] - (df1['CRuns'] / df1['Years'])
df1['NEW_Diff_RBI'] = df1['RBI'] - (df1['CRBI'] / df1['Years'])
df1['NEW_Diff_Walks'] = df1['Walks'] - (df1['CWalks'] / df1['Years'])

def feature_ext(df1):
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    new_num_cols = [col for col in num_cols if col != "Salary"]
    df1[new_num_cols] = df1[new_num_cols] + 0.0000000001

    df1['NEW_Hits'] = df1['Hits'] / df1['CHits'] + df1['Hits']
    df1['NEW_RBI'] = df1['RBI'] / df1['CRBI']
    df1['NEW_Walks'] = df1['Walks'] / df1['CWalks']
    df1['NEW_PutOuts'] = df1['PutOuts'] * df1['Years']
    df1["Hits_Success"] = (df1["Hits"] / df1["AtBat"]) * 100
    df1["NEW_CRBI*CATBAT"] = df1['CRBI'] * df1['CAtBat']
    df1["NEW_RBI"] = df1["RBI"] / df1["CRBI"]
    df1["NEW_Chits"] = df1["CHits"] / df1["Years"]
    df1["NEW_CHmRun"] = df1["CHmRun"] * df1["Years"]
    df1["NEW_CRuns"] = df1["CRuns"] / df1["Years"]
    df1["NEW_Chits"] = df1["CHits"] * df1["Years"]
    df1["NEW_RW"] = df1["RBI"] * df1["Walks"]
    df1["NEW_RBWALK"] = df1["RBI"] / df1["Walks"]
    df1["NEW_CH_CB"] = df1["CHits"] / df1["CAtBat"]
    df1["NEW_CHm_CAT"] = df1["CHmRun"] / df1["CAtBat"]
    df1['NEW_Diff_Atbat'] = df1['AtBat'] - (df1['CAtBat'] / df1['Years'])
    df1['NEW_Diff_Hits'] = df1['Hits'] - (df1['CHits'] / df1['Years'])
    df1['NEW_Diff_HmRun'] = df1['HmRun'] - (df1['CHmRun'] / df1['Years'])
    df1['NEW_Diff_Runs'] = df1['Runs'] - (df1['CRuns'] / df1['Years'])
    df1['NEW_Diff_RBI'] = df1['RBI'] - (df1['CRBI'] / df1['Years'])
    df1['NEW_Diff_Walks'] = df1['Walks'] - (df1['CWalks'] / df1['Years'])
    return df1




#############################################
# 4. One-Hot Encoding
#############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df1)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool':
            dataframe[col] = dataframe[col].astype(int)
    return dataframe

df1 = one_hot_encoder(df1, cat_cols, drop_first=True)


#############################################
# 5. Feature Scaling (Özellik Ölçeklendirme)
#############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df1)

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df1[num_cols] = scaler.fit_transform(df1[num_cols])
df1.head()


def feature_scaling(dataframe, num_cols):
    # Özellik ölçeklendirme işlemleri
    scaler = StandardScaler()
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if col not in ["Salary"]]
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe

# Correlation Analysis
fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(df1.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()


#############################################
#               MODELING                    #
#############################################
df1.isnull().sum().sum()
#df1.dropna(inplace=True)
y = df1["Salary"]
X = df1.drop(["Salary"], axis=1)

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=46)

# Model Evaluation for Linear Regression
linreg = LinearRegression()
model = linreg.fit(X_train,y_train)
y_pred = model.predict(X_train)
lin_train_rmse =np.sqrt(mean_squared_error(y_train,y_pred))
print("LINEAR REGRESSION TRAIN RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train,y_pred))))

lin_train_r2 = linreg.score(X_train,y_train)
print("LINEAR REGRESSION TRAIN R-SQUARED:", "{:,.3f}".format(linreg.score(X_train,y_train)))

y_pred = model.predict(X_test)
lin_test_rmse =np.sqrt(mean_squared_error(y_test,y_pred))
print("LINEAR REGRESSION TEST RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test,y_pred))))

lin_test_r2 = linreg.score(X_test,y_test)
print("LINEAR REGRESSION TEST R-SQUARED:", "{:,.3f}".format(linreg.score(X_test,y_test)))


# Test part regplot:
g = sns.regplot(x=y_test, y=y_pred, scatter_kws={'color': 'b', 's': 5},
                ci=False, color="r")
g.set_title(f"Test Model R2: = {linreg.score(X_test, y_test):.3f}")
g.set_ylabel("Predicted Salary")
g.set_xlabel("Salary")
plt.xlim(-5, 2700)
plt.ylim(bottom=0)
plt.show(block=True)


# Cross Validation Score
print("LINEAR REGRESSION CROSS_VAL_SCORE:", "{:,.3f}".format(np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))))

# Bagimsiz degiskenin bagimli degiskene etkisini görebiliyoruz?
#

# OLS for Linear Regression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Adding a constant to the model (necessary for statsmodels)
X_train_sm = sm.add_constant(X_train)

# Fitting the model using statsmodels
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Getting the summary of the regression model
model_summary = model_sm.summary()
model_summary

def model_training(dataframe,target_col):
    y = dataframe[target_col]
    X = dataframe.drop([target_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=46)

    # Model Evaluation for Linear Regression
    linreg = LinearRegression()
    model = linreg.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    lin_train_rmse =np.sqrt(mean_squared_error(y_train,y_pred))
    print("LINEAR REGRESSION TRAIN RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train,y_pred))))

    lin_train_r2 = linreg.score(X_train,y_train)
    print("LINEAR REGRESSION TRAIN R-SQUARED:", "{:,.3f}".format(linreg.score(X_train,y_train)))

    y_pred = model.predict(X_test)
    lin_test_rmse =np.sqrt(mean_squared_error(y_test,y_pred))
    print("LINEAR REGRESSION TEST RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test,y_pred))))

    lin_test_r2 = linreg.score(X_test,y_test)
    print("LINEAR REGRESSION TEST R-SQUARED:", "{:,.3f}".format(linreg.score(X_test,y_test)))
    # Adding a constant to the model (necessary for statsmodels)
    #X_train_sm = sm.add_constant(X_train)
    # Fitting the model using statsmodels
    #model_sm = sm.OLS(y_train, X_train_sm).fit()
    # Getting the summary of the regression model
    #model_summary = model_sm.summary()
    #return model_summary


##### Functions ######

def sonuc(df,method):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df1 = eksik_veri_doldur(df,method)
    df1 = feature_ext(df1)
    cat_cols, num_cols, cat_but_car = grab_col_names(df1)
    df1 = one_hot_encoder(df1, cat_cols, drop_first=True)
    df1 = feature_scaling(df1, num_cols)
    model = model_training(df1,"Salary")

sonuc(df,method=1)

#############################################
# All Models
#############################################
# Gelismis Agac Yöntemleri sonrasinda bu alana devam edebilirsiniz.
# Keyifli calismalar...

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(verbose=-1)),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



################################################
# Random Forests
################################################

rf_model = RandomForestRegressor(random_state=17)

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

################################################
# GBM Model
################################################

gbm_model = GradientBoostingRegressor(random_state=17)

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

################################################
# LightGBM
################################################

lgbm_model = LGBMRegressor(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [300, 500],
                "colsample_bytree": [0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

################################################
# CatBoost
################################################

catboost_model = CatBoostRegressor(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse


######################################################
#  Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}


lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


regressors = [("RF", RandomForestRegressor(), rf_params),
              ('GBM', GradientBoostingRegressor(), gbm_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              ("CatBoost", CatBoostRegressor(), catboost_params)]


best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
