from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from src.models.base_model import BaseModel


class LDAModel(BaseModel):
    def __init__(self):
        super().__init__(
            model=LinearDiscriminantAnalysis(
            )
        )
