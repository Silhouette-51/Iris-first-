import numpy as np
import matplotlib.pyplot as plt

#定义决策树节点类
class Node:
    def __init__(self,predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self,max_depth=None,max_features=None,random_state=None):
        self.max_depth = max_depth #决策树最大深度
        self.max_features = max_features #决策树节点最大特征数
        self.random_state = random_state
        self.tree = None

    def impurity_measure(self,y):
        unique_classes,counts = np.unique(y,return_counts=True)
        total_samples = len(y)
        impurity = 1
        for count in counts:
            p = count/total_samples
            impurity-=p**2
        return impurity

    #构建决策树
    def fit(self,x,y):
        self.n_classes = len(set(y))
        self.n_features = x.shape[1]

        if self.max_features is None:
            self.max_features = self.n_features

        if 0<self.max_features <= 1 and isinstance(self.max_features,float):
            self.max_features = int(self.max_features*self.n_features)

        self.tree = self.grow_tree(x,y,self.random_state)

    def split_data(self,x,y,random_state):
        m = len(y)
        if m<1:
            return None
        impurity_base = self.impurity_measure(y)
        if impurity_base == 0:
            return None,None

        best_index,best_threshold = None,None

        np.random.seed(random_state)
        feature_indices = np.random.choice(range(self.n_features),self.max_features,replace=False)

        for feature_idx in feature_indices:
            feature_values = sorted(set(x[:,feature_idx]))
            possible_features = [np.mean([i,j]) for i,j in zip(feature_values,feature_vlaues[1:])]

            for threshold in possible_features:
                left_y = y[x[:,feature_idx]<threshold]
                right_y = y[x[:,feature_idx]>=threshold]

                left_ratio , right_ratio = len(left_y)/m , len(right_y)/m
                impurity_current = left_ratio * self.impurity_measure(left_y) + right_ratio * self.impurity_measure(right_y)

                if impurity_current > impurity_base:
                    best_index = feature_idx
                    best_threshold = threshold
                    impurity_base = impurity_current

            return best_index, best_threshold

    def grow_tree(self,x,y,random_state,depth=0):


