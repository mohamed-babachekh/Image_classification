import numpy as np
import itertools
import matplotlib.pyplot as plt
from src.models.base_model import BaseModel
from src.constants import MODEL_PATH
import os

class PlotUtils:
    @staticmethod
    def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        title = 'Confusion Matrix of {}'.format(title)

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    @staticmethod
    def plot_kfold_cross_validation(title,results,names):
        title = 'Confusion Matrix of {}'.format(title)
        # fig = plt.figure()
        plt.title(title)
        # fig.suptitle('Machine Learning algorithm comparison')
        # ax = fig.add_subplot(111)
        plt.boxplot(results)
        # ax.set_xticklabels(names)
        plt.xlabel(names)
        plt.tight_layout()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')


def change_output_format(option):
    return option[1]

def infere(features,model_name):
    model=BaseModel()
    model.load(MODEL_PATH.replace('aggregator_model',model_name))
    prediction = model.predict(features.reshape(1,-1))[0]

    
    return prediction


def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
  
        # Checking if the directory is empty or not
        if not os.listdir(path):
            print("Empty directory")
        else:
            print("Not empty directory")
    else:
        print("The path is either for a file or not valid")
  