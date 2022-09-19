############################################
# ASSOCIATION RULE LEARNING
############################################
## outliers functions

def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


## information functions

def check_df(dataframe, head=5):
    print("##   Shape    ##")
    print(dataframe.shape)
    print("##   Types    ##")
    print(dataframe.dtypes)
    print("##   Head   ##")
    print(dataframe.head(head))
    print("##   Tail   ##")
    print(dataframe.tail(head))
    print("##   Missing entries   ##")
    print(dataframe.isnull().sum())
    print("##   Quantiles   ##")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##   general information   ##")
    print(dataframe.describe().T)


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


##Problem related functions

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, cr=None, id=False):
    if cr != None:
        dataframe = dataframe[dataframe["Country"] == cr]
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

############################################
# 1. Data Preprocessing
############################################

import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_excel("online_retail_II/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()
df.head()
check_df(df)
#
# ##   Shape    ##
# (405633, 8)
# ##   Types    ##
# Invoice                object
# StockCode              object
# Description            object
# Quantity                int64
# InvoiceDate    datetime64[ns]
# Price                 float64
# Customer ID           float64
# Country                object
# dtype: object
# ##   Head   ##
#   Invoice StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID         Country
# 0  536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6 2010-12-01 08:26:00   2.55      17850.0  United Kingdom
# 1  536365     71053                  WHITE METAL LANTERN         6 2010-12-01 08:26:00   3.39      17850.0  United Kingdom
# 2  536365    84406B       CREAM CUPID HEARTS COAT HANGER         8 2010-12-01 08:26:00   2.75      17850.0  United Kingdom
# 3  536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6 2010-12-01 08:26:00   3.39      17850.0  United Kingdom
# 4  536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6 2010-12-01 08:26:00   3.39      17850.0  United Kingdom
# ##   Tail   ##
#        Invoice StockCode                      Description  Quantity         InvoiceDate  Price  Customer ID Country
# 541904  581587     22613      PACK OF 20 SPACEBOY NAPKINS        12 2011-12-09 12:50:00   0.85      12680.0  France
# 541905  581587     22899     CHILDREN'S APRON DOLLY GIRL          6 2011-12-09 12:50:00   2.10      12680.0  France
# 541906  581587     23254    CHILDRENS CUTLERY DOLLY GIRL          4 2011-12-09 12:50:00   4.15      12680.0  France
# 541907  581587     23255  CHILDRENS CUTLERY CIRCUS PARADE         4 2011-12-09 12:50:00   4.15      12680.0  France
# 541908  581587     22138    BAKING SET 9 PIECE RETROSPOT          3 2011-12-09 12:50:00   4.95      12680.0  France
# ##   Missing entries   ##
# Invoice        0
# StockCode      0
# Description    0
# Quantity       0
# InvoiceDate    0
# Price          0
# Customer ID    0
# Country        0
# dtype: int64
# ##   Quantiles   ##
#                 0.00      0.05      0.50     0.95      0.99     1.00
# Quantity    -80995.0      1.00      5.00     36.0    120.00  80995.0
# Price            0.0      0.42      1.95      8.5     14.95  38970.0
# Customer ID  12346.0  12630.00  15159.00  17908.0  18212.00  18287.0
# ##   general information   ##
#                 count          mean          std      min       25%       50%       75%      max
# Quantity     405633.0     12.089465   249.059161 -80995.0      2.00      5.00     12.00  80995.0
# Price        405633.0      3.358961    66.980347      0.0      1.25      1.95      3.75  38970.0
# Customer ID  405633.0  15294.737566  1710.302540  12346.0  13969.00  15159.00  16794.00  18287.0

# We can see that the price and quantity have an outliers problem, as the values from the quantiles show.
# The values at 99% of the column is around 120 for quantity while it is around 81k at 100%.
# Same goes for the price as its 99% values are around 15 while 100 increases up to 39k.
# Another problem is we have negative values in quantity which is not normal,
# This might be coming from the cancelled invoices

# Drop POST StockCodes
df = df[df["StockCode"] != "POST"]

# Checking for missing Values and dropping them
df.isnull().sum()
df.dropna(inplace=True)

# Invoices with C in their values indicates that the invoice was cancelled.
# we will remove them the dataset, as well as remove the negative values from quantity and price

df = retail_data_prep(df)

## Outliers
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

# Quantity True
# InvoiceDate False
# Price True
# Customer ID False

# replace them
for col in num_cols:
    replace_with_thresholds(df, col, q1=0.05, q3=0.95)

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

# Quantity False
# InvoiceDate False
# Price False
# Customer ID False


############################################
# 2. Invoice-Product Matrix Creation
############################################
df_GR = df[df['Country'] == "Germany"]
df_GR_MAT = create_invoice_product_df(df,"Germany",True)
df_GR_MAT.head()


############################################
# 3. Create association rules
############################################


frequent_itemsets = apriori(df_GR_MAT,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

############################################
# 5. Recommend a product
############################################
# Let's take a random product as an example:

product_id = 20724
check_id(df, product_id)

# Our random product is RED RETROSPOT CHARLOTTE BAG

# To get 3 product recommendation we can use the arl_recommender function to get recommendations:
arl_recommender(rules, 22492, 3)
# [21915, 22328, 22331]

check_id(df, 21915) # RED  HARMONICA IN BOX
check_id(df, 22328) # ROUND SNACK BOXES SET OF 4 FRUITS
check_id(df, 22331) # WOODLAND PARTY BAG + STICKER SET


df_GR.head()