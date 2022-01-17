from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel
from src.constants import NUM_TREES,SEED

class RandomForestModel(BaseModel):
    def __init__(self, n_jobs: int=4,random_state:int=SEED ,criterion:str='gini', n_estimators:int=NUM_TREES):
        self.n_jobs=n_jobs
        self.random_state=random_state
        self.criterion=criterion
        self.n_estimators=n_estimators


        super().__init__(
            model=RandomForestClassifier(
            n_jobs=self.n_jobs,
            random_state= self.random_state,
            criterion= self.criterion,
            n_estimators= self.n_estimators,
            )
        )
