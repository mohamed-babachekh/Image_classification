from sklearn.neighbors import KNeighborsClassifier

from src.models.base_model import BaseModel


class KNNModel(BaseModel):
    def __init__(self):
        super().__init__(
            model=KNeighborsClassifier(
            )
        )
