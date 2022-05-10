import numpy as np
from matplotlib import pyplot as plot

STUDENT_SIZE = 100


def generate_students_info():
    wt = np.random.uniform(low=40.0, high=90.0, size=STUDENT_SIZE)
    ht = np.random.randint(low=140, high=200, size=STUDENT_SIZE)
    bmi = wt / (ht * ht) * 10000
    return (wt, ht, bmi)


def bmifilter(bmi):
    label = ["Underweight", "healty", "Overweight", "Obese"]
    dist = np.zeros(4, dtype="int64")
    for item in np.nditer(bmi):
        if item < 25:
            index = 0 if item < 18.5 else 1
        else:
            index = 2 if item < 30 else 3
        dist[index] += 1
    return dist, label


def bar(bmi, label):
    plot.bar(label, bmi)
    plot.show()


def hist(bmi):
    bins = [0, 18.5, 25, 30, max(bmi)]
    plot.hist(bmi, bins)
    plot.xlabel("bmi")
    plot.xticks(bins)
    plot.show()


def pie(bmi, label):
    plot.pie(bmi, labels=label, autopct="%1.2f%%")
    plot.show()
    pass


def scat(wt, ht):
    plot.scatter(ht, wt)
    plot.xlabel("height")
    plot.ylabel("weight")
    plot.show()


if __name__ == "__main__":
    wt, ht, bmi = generate_students_info()
    d_bmi, label = bmifilter(bmi)
    bar(d_bmi, label)
    hist(bmi)
    pie(d_bmi, label)
    scat(wt, ht)
