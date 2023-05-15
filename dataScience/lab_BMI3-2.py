from typing import List, Any, Tuple
from enum import Enum
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Define Keys for Easy handling
Height = "Height (Inches)"
Weight = "Weight (Pounds)"


# Define Group Enum
class Group(Enum):
    Null = 0
    Gender = 1
    BMI = 2


# Define data_loader with closure
def data_loader(path: str):
    # Load DataFrame From Path
    loaded = pd.read_csv(path)
    # Region Dirty Data of Weight and Height To NA
    # Height under 10 or over 1000
    loaded.loc[500 < loaded[Height], Height] = np.nan
    loaded.loc[10 > loaded[Height], Height] = np.nan
    # Weight under 20 or over 500
    loaded.loc[500 < loaded[Weight], Weight] = np.nan
    loaded.loc[20 > loaded[Weight], Weight] = np.nan
    # Drop botof Geight and Weight is na
    loaded.dropna(subset=[Weight, Height], how="all", inplace=True)
    # endregion

    # data provider closure
    def get_datas(group: Group) -> Tuple[List[pd.DataFrame], List[str]]:
        # get copy of datafrme to prevant Duplicate IO from 2nd Disk
        df = loaded.copy()

        # Divide DataFrame by Group
        group_column: str = ""
        if group == Group.Null:
            return ([df], [""])
        elif group == Group.Gender:
            group_column = "Sex"
        elif group == Group.BMI:
            group_column = "BMI"
        # drop na with make labels
        group_list = df[group_column].unique()
        group_list = group_list[~pd.isna(group_list)]
        keys: List[str] = [item for item in group_list.tolist()]
        return ([df[df[group_column] == key] for key in keys], keys)

    return get_datas


# Function for LinearLigression
def regression(df: pd.DataFrame):
    def directional_regression(
        start: pd.Series, to: pd.Series, req_predit: pd.Series
    ) -> np.ndarray:
        regressor = LinearRegression()
        regressor.fit(start[:, np.newaxis], to)
        pred = regressor.predict(req_predit[:, np.newaxis])
        ret = pd.DataFrame(pred, columns=["value"], index=req_predict.index.tolist())
        return ret["value"]

    # gen na-less Dataframe
    for_train = df.copy()
    for_train.dropna(subset=[Weight, Height], how="any", inplace=True)
    weight: pd.Series = for_train[Weight]
    height: pd.Series = for_train[Height]

    # Region Repace Na of Height by regression
    if weight.shape[0] != 0:
        req_predict = df.dropna(subset=[Weight], how="any")[Weight]
        predicted = directional_regression(weight, height, req_predict)
        # eventhou drop, index is kept
        for index in df[df[Height].isna()].index.tolist():
            df[Height][index] = predicted[index]
    # endregion

    # Region Repace Na of Weight by regression
    if height.shape[0] != 0:
        req_predict = df.dropna(subset=[Height], how="any")[Height]
        predicted = directional_regression(height, weight, req_predict)
        # eventhou drop, index is kept
        for index in df[df[Weight].isna()].index.tolist():
            df[Weight][index] = predicted[index]
    # endregion
    return df


# Funcrion for show Plt
def show_plt(title: str, df: List[pd.DataFrame], labels: List[str], color: List[str]):
    for index in range(df.__len__()):
        plt.scatter(
            x=df[index][Weight].to_numpy(),
            y=df[index][Height].to_numpy(),
            color=color[index],
            label=labels[index],
        )
    plt.xlabel(Weight)
    plt.ylabel(Height)
    plt.legend()
    plt.title(title)
    plt.show()


# control function for regression, show plot
def control(title: str, group: Group, color: List[str]):
    reg: List[pd.DataFrame] = []
    df, labels = load(group)
    print(labels)
    for index in range(df.__len__()):
        reg.append(regression(df[index]))
    show_plt(title, reg, labels, plt_colors)


load = data_loader("./bmi_data_lab3.csv")

# predefine for plt oolors
plt_colors = ["Red", "Green", "Blue", "Black"]


# Plt for not grouped

control("Not Grouped", Group.Null, plt_colors)

# plt for Gender Grouped
control("Gender Grouped", Group.Gender, plt_colors)

# plt for BMI Grouped
control("BMI Grouped", Group.BMI, plt_colors)

