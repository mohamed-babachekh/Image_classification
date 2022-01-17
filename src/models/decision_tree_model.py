from sklearn.tree import DecisionTreeClassifier

from src.models.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, random_state: int = 9):
        self.random_state = random_state
     

        super().__init__(
            model=DecisionTreeClassifier(
                random_state=self.random_state
                
            )
        )
