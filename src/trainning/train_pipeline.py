from os import name
from src.constants import H5_DATA,H5_LABELS, NUM_TREES, SEED, TEST_SIZE, MODEL_PATH,SCORING, VALIDATION_PATH
import numpy as np
import h5py
from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.models.logistic_regression_model import LogisticRegression
from src.models.LDA_model import LDAModel
from src.models.KNN_model import KNNModel
from src.models.decision_tree_model import DecisionTreeClassifier, DecisionTreeModel
from src.models.random_forest_model import RandomForestClassifier, RandomForestModel
from src.models.naive_bais_model import NaiveBayesModel
from src.models.svc_model import SVCModel
from src.models.base_model import BaseModel
from src.utils import PlotUtils

import matplotlib.pyplot as plt

from sklearn import linear_model 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class TrainingPipeline:
    def __init__(self):
        h5f_data  = h5py.File(H5_DATA, 'r')
        h5f_label = h5py.File(H5_LABELS, 'r')

        global_features_string = h5f_data['dataset_1']
        global_labels_string   = h5f_label['dataset_1']

        global_features = np.array(global_features_string)
        global_labels   = np.array(global_labels_string)

        h5f_data.close()
        h5f_label.close()
# verify the shape of the feature vector and labels
        print("[STATUS] features shape: {}".format(global_features.shape))
        print("[STATUS] labels shape: {}".format(global_labels.shape))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(global_features),
                                                                                np.array(global_labels),
                                                                                test_size=TEST_SIZE,
                                                                                random_state=SEED
                                                                                )

        self.model = None

        self.models=[]
        self.models.append(('LR', LogisticRegression()))
        self.models.append(('LDA', LDAModel()))
        self.models.append(('KNN', KNNModel()))
        self.models.append(('CART', DecisionTreeModel()))
        self.models.append(('RF', RandomForestModel()))
        self.models.append(('NB', NaiveBayesModel()))
        self.models.append(('SVM', SVCModel()))


    def  train(self ,model_name: str = 'model.joblib'):
        for name,model in self.models:
            if model_name== name:
                self.model=model
            else:
                self.model=RandomForestModel()
                
        self.model.fit(x_train = self.x_train,
                    y_train = self.y_train)
        print('model trained',model_name)
        model_path = str(MODEL_PATH).replace('aggregator_model', model_name)
        
        BaseModel.save(self.model,model_path)

    def get_model_perfomance(self) -> tuple:
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions), f1_score(self.y_test, predictions,average='weighted')

    def cross_validation(self,plot_name: str = 'plot.png'):
        # variables to hold the results and names
        results = []
        names   = []
        models = []
        models.append(('LR',linear_model.LogisticRegression(random_state=SEED)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier(random_state=SEED)))
        models.append(('RF', RandomForestClassifier(n_estimators=NUM_TREES, random_state=SEED)))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(random_state=SEED)))
        for name, model in models:
            kfold = KFold(n_splits=10)
            
            cv_results = cross_val_score(model, self.x_train, self.y_train, cv=kfold, scoring=SCORING,)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        PlotUtils.plot_kfold_cross_validation(
            title='10 Fold Cross Validation',
            results=results,
            names=names
        )
        plot_path = str(VALIDATION_PATH).replace('cm_plot', plot_name)
        plt.savefig(plot_path,bbox_inches='tight',pad_inches=0.33)
        



if __name__ == "__main__":
    tp = TrainingPipeline()
    accuracy, f1 = tp.get_model_perfomance()
    tp.render_confusion_matrix()
    print("ACCURACY = {}, F1 = {}".format(accuracy, f1))
