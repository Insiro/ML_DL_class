# set able type hinting self embeding class
from __future__ import annotations
# type hint library
from typing import Any, Tuple
import pandas as pd
from math import log2

# region set global varible for reduce typeing error
TARGET = "Outcome"
FEATURES = ["District", "HosueType", "Income", "PerviousCustomer"]
# endregion


class DecisionNode:
    triggers: list[Any]

    def __init__(self, df: pd.DataFrame, features: list[str], feature: str):
        # self containing dataFrame
        self.df = df
        # feature that devide value
        self.feature = feature
        # feature list for pass to next
        self.features = features
        # values to select child
        if self.features.__len__ == 0:
            self.triggers = [True]
        else:
            self.triggers = df[feature].unique().tolist()
        self.triggers.sort()
        # probability that true when predict value is same with trigger
        self.probabilities: list[float] = []
        self.entropy = self.__calculateEntropy()
        # invalid informationGain
        self.informationGain: float = -1
        self.childs: list[DecisionNode] = []

    # calculate entorpy
    def __calculateEntropy(self) -> float:
        entropy = 0
        # get entropy of each values
        for value in self.triggers:
            # filter datafreame by value
            filtered = self.df[self.df[self.feature] == value]
            # get number of true in filtered
            true_count = filtered[filtered[TARGET]].count()[0]
            # region partial entropy calc
            filtered_count = filtered.count()[0]
            if (true_count == 0):  # the case of probability is 0
                self.probabilities.append(0)
                continue
            probability = true_count/filtered_count
            self.probabilities.append(probability)
            # log(A/B) = Log A- B, for make simple log caluculation devided
            entropy -= probability * (log2(true_count) - log2(filtered_count))
            # endregion
        return entropy

    def select_child(self):
        # leaf node case
        if self.features.__len__() == 0:
            return
        # turn information_gain into valid
        self.informationGain = 0
        # child unit calc information gain
        for value in self.triggers:
            # data frame that filtetered by This node
            next_df = self.df[self.df[self.feature] == value]
            # region select next node that having smallest entopy
            candidateNode: DecisionNode
            candidateNodeEntropy = float("inf")
            for feature in self.features:
                # filtering nextFeatures for not containing self feature
                subFeatures = [
                    f for f in self.features if f != feature]
                node = DecisionNode(next_df, subFeatures, feature)
                node_entropy = node.entropy
                if candidateNodeEntropy > node_entropy:
                    candidateNode = node
                    candidateNodeEntropy = node_entropy
            self.childs.append(candidateNode)
            self.informationGain += node_entropy
            # endregion
        for child in self.childs:
            child.select_child()
        return

    def predict(self, record: dict[str, Any]) -> Tuple[Any, float]:
        if record[self.feature] not in self.triggers:
            # get case of probability is most far from 0.5
            index = self.probabilities.index(max(self.probabilities))
            # when not case in data, return most probability
            if self.features.__len__() == 0:
                if self.probabilities[index] > 0.5:
                    return (True, self.probabilities[index])
                else:
                    return (False, 1-self.probabilities[index])
            return self.childs[index].predict(record)
        caseIndex = self.triggers.index(record[self.feature])
        # when no case in data go to most probability
        if self.features.__len__() == 0:
            # case True
            if self.probabilities[caseIndex] > 0.5:
                return (True, self.probabilities[caseIndex])
            # case False, return probability of false
            else:
                return (False, 1-self.probabilities[caseIndex])
        return self.childs[caseIndex].predict(record)

    def print_tree(self, depth, trigger):
        shift = ""
        for i in range(depth):
            shift += "\t"
        print(shift + self.feature, trigger)
        for i in range(self.childs.__len__()):
            self.childs[i].print_tree(depth+1, self.triggers[i])


def generateDecisionTree(df: pd.DataFrame):
    # region select Root Node by Stemp Information gain
    gains: list[float] = []
    roots: list[DecisionNode] = []
    for feature in FEATURES:
        # get feature list that not containing node feature
        subFeatures = [f for f in FEATURES if f != feature]
        Root = DecisionNode(df, subFeatures, feature)
        gain = Root.informationGain
        gains.append(gain)
        roots.append(Root)
        break

    root = roots[gains.index(min(gains))]
    # endregion
    # select childs for build tree
    root.select_child()
    return root


if __name__ == '__main__':
    # region Generage DataFrame
    df = pd.DataFrame()
    df[FEATURES[0]] = ["Suburban", "Suburban", "Rural", "Urban", "Urban", "Urban",
                       "Rural", "Suburban", "Suburban", "Urban", "Suburban", "Rural", "Rural", "Urban"]
    df[FEATURES[1]] = ["Detached", "Detached", "Detached", "Semi-detached", "Semi-detached", "Semi-detached",
                       "Semi-detached", "Terrace", "Semi-detached", "Terrace", "Terrace", "Terrace", "Detached", "Terrace"]
    df[FEATURES[2]] = ["High", "High", "High", "High", "Low", "Low",
                       "Low", "High", "Low", "Low", "Low", "High", "Low", "High"]
    # True : Yes
    df[FEATURES[3]] = [False, True, False, False, False,
                       True, True, False, False, False, True, True, False, True]
    # Responsed : True
    df[TARGET] = [False, False, True, True, True, False,
                  True, False, True, True, True, True, True, False]
    # endregion

    tree = generateDecisionTree(df)
    # testing
    tree.print_tree(0, "")
    predicted, probability = tree.predict(
        {FEATURES[0]: "Suburban", FEATURES[1]: "Detached", FEATURES[2]: "High", FEATURES[3]: False})
    print("{}% {}".format(probability * 100, predicted))
