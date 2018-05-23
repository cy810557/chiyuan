from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
def blind_grid_search(model, X, y):
    # C_range = np.logspace(-1,5,5)
    C_range = np.linspace(0.1,0.3,10)
    gamma_range = np.logspace(0,1,20)
    param_grid = dict(gamma = gamma_range, C = C_range)
    grid = GridSearchCV(model, param_grid)
    grid.fit(X, y)
    print(
        'The best parameters are {} with a score of {:0.2f}.'.format(
        grid.best_params_,grid.best_score_))
iris = load_iris()
feature = iris.data
label = iris.target
my_classifier = SVC()
def visualize_grid_search(model,X, y):
    C_range = np.logspace(-2,1,5)
    gamma_range = np.logspace(0,2,5)
    param_grid = dict(gamma = gamma_range, C = C_range)
    grid = GridSearchCV(model, param_grid)
    grid.fit(X, y)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    #设置热力图(②也可以用自带的热力图：cmap = plt.cm.Greens ③plot.cm.get_cmap())
    ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91', \
                '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
    ddlheatmap = colors.ListedColormap(ddl_heat)
    cm = plt.cm.get_cmap('RdYlBu')

    plt.figure(figsize=(8,6))
    plt.subplots_adjust(left = .2,right = .95, bottom = .15, top = .95)
    # plt.imshow(scores, interpolation='nearest', cmap=ddlheatmap)  #高级操作
    # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues)
    plt.imshow(scores, interpolation='nearest', cmap=cm)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(
        "The best parameters are {} with a score of {:0.2f}.".format(
        grid.best_params_, grid.best_score_)
    )
    plt.show()
visualize_grid_search(my_classifier, feature, label)



