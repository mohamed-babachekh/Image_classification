from sklearn.svm import SVC

from src.models.base_model import BaseModel
from src.constants import SEED

class SVCModel(BaseModel):
    def __init__(self ,C: int=10):
        # self.random_state= random_state,
        self.C=C,
        # self.gamma=gamma,
        # self.kernel=kernel

        super().__init__(
            model=SVC(
            # random_state = self.random_state,
            C=self.C,
            # gamma= self.gamma,
            # kernel=self.kernel
            )
        )


# 'C': 10, 'gamma': 'scale', 'kernel': 'rbf'
