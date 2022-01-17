from sklearn.linear_model import LogisticRegression
from src.constants import SEED
from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, C: float=4.122186038401156,penalty:str="l2", solver:str="newton-cg"):
        # self.random_state=random_state
        self.C=C
        self.penalty=penalty
        self.solver=solver

        super().__init__(
            model=LogisticRegression(
                # random_state=self.random_state,
                C=self.C,
                penalty=self.penalty,
                solver=self.solver,

            )
        )
