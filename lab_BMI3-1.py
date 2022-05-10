import pandas as pd
import numpy as np
import sklearn as skl
from matplotlib import pyplot as plt




loaded = pd.read_csv("./bmi_data_lab3.csv")
print(loaded.info())
print("Sex Keys", loaded["Sex"].unique())

# Define Keys for Easy handling
Height = "Height (Inches)"
Weight = "Weight (Pounds)"

# Get BMI list
BMI_bins = loaded["BMI"].unique()
BMI_bins = BMI_bins[~pd.isna(BMI_bins)]
BMI_bins = BMI_bins.tolist()
# Devide BMI
datas = {}
for item in BMI_bins:
    datas.update({item: loaded[loaded["BMI"] == item]})
# predefine color for hisogram
colors = ["red", "green", "blue", "yellow"]
# Print Histograms of Height
i = 0
for key in datas.keys():
    plt.hist(datas[key][Height], bins=10, color=colors[i], label="BMI : " + str(key))
    plt.title("Height")
    i += 1
plt.legend()
plt.show()

# Print Histograms of Weight
i = 0
for key in datas.keys():
    plt.hist(datas[key][Weight], bins=10, color=colors[i], label="BMI : " + str(key))
    plt.title("Weight")
    i += 1
plt.legend()
plt.show()