import pandas as pd

# region set global varible for reduce typeing error
HEIGHT = "HEIGHT"
WEIGHT = "WEIGHT"
TARGET = "SIZE"
M = "M"
L = "L"
# endregion


class KNN:
    def __init__(self, N: int, df: pd.DataFrame):
        self.n = N
        self.df = df
        self.MinMaxScale(HEIGHT)
        self.MinMaxScale(WEIGHT)

    def MinMaxScale(self, feature: str):
        min = self.df[feature].max()
        divider = self.df[feature].max() - min
        self.df[feature + "_scaled"] = (self.df[feature] - min) / divider

    def predict(self, height: int, weight: int):
        distance = "distance"
        self.df[distance] = (
            (self.df[HEIGHT] - height) * (self.df[HEIGHT] - height)
            + (self.df[WEIGHT] - weight)
            + (self.df[WEIGHT] - weight)
        )
        self.df = self.df.sort_values(by=[distance])
        head = self.df[0 : self.n]
        return head[TARGET].mode()[0]


# region generate dataframe
df = pd.DataFrame()
df[HEIGHT] = [
    158,
    158,
    158,
    160,
    160,
    163,
    163,
    160,
    163,
    165,
    165,
    165,
    168,
    168,
    168,
    170,
    170,
    170,
]
df[WEIGHT] = [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68]
df[TARGET] = [M, M, M, M, M, M, M, L, L, L, L, L, L, L, L, L, L, L]
# endregion
# test
knn = KNN(5, df)
print(knn.predict(158, 58))
