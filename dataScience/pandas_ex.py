import numpy as np
import pandas as pd

# region Data define
datas = np.array(
    [3.0, "?", 2.0, 5.0, "*", 4.0, 5.0, 6.0, "+", 3.0, 2.0, "&", 5.0, "?", 7.0, "!"],
    dtype=str,
)
datas = datas.reshape(4, 4)
DataFrame = pd.DataFrame(datas)
DataFrame.replace("^.{1}$", pd.NA, regex=True, inplace=True)
DataFrame = DataFrame.apply(pd.to_numeric)
# endregion
print(DataFrame)
print("isNA with any")
print(DataFrame.isna().any())
print("isNA with sum")
print(DataFrame.isna().sum())
print("drop NA with any")
print(DataFrame.dropna(how="any"))
print("drop NA with all")
print(DataFrame.dropna(how="all"))
print("drop NA with thresh 1")
print(DataFrame.dropna(thresh=1))
print("drop NA with thresh 2")
print(DataFrame.dropna(thresh=2))
print("fill na with 100")
print(DataFrame.fillna(100))
print("fill na with mean")
print(DataFrame.fillna(DataFrame.mean()))
print("fill na with median")
print(DataFrame.fillna(DataFrame.median()))
print("ffill")
print(DataFrame.ffill())
print("bfill")
print(DataFrame.bfill())
