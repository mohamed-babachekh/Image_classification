from sklearn.naive_bayes import GaussianNB


from src.models.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    def __init__(self):
        super().__init__(
            model=GaussianNB(
            )
        )
