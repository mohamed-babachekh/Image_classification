#--------------------
# tunable-parameters
#--------------------

IMAGE_PER_CLASS = 80
FIXED_SIZE = tuple((500, 500))
DATASET_PATH="dataset/jpg"
TRAIN_PATH = "dataset/train"
H5_DATA = 'output/data.h5'
H5_LABELS =  'output/labels.h5'
BINS = 8

NUM_TREES = 100
TEST_SIZE = 0.10
SEED      = 9
TEST_PATH  = "dataset/test"
SCORING    = "accuracy"


MODEL_PATH= 'output/models/aggregator_model.joblib'

VALIDATION_PATH='output/plot/cm_plot'
PREDICTION_PATH='output/plot/infere'

MODELS=[]
MODELS.append(('LR', 'Logistic Regression'))
MODELS.append(('LDA', 'Linear Discriminant Analysis '))
MODELS.append(('KNN', 'K-Nearest Neighbor '))
MODELS.append(('CART', 'Decision Tree Classifier '))
MODELS.append(('RF', 'Random Forest Classifier '))
MODELS.append(('NB', 'Naive Bayes '))
MODELS.append(('SVM', 'SVM '))