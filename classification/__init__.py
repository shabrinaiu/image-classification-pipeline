from utils.types import Classification


class BaseClassifier:
    class_names = [
        "_background_",
        "All_New_Daihatsu_Terios",
        "Toyota_Agya",
        "Toyota_Avanza",
        "Toyota_Fortuner",
        "Honda_Brio",
        "Honda_CRV",
        "Honda_Jazz",
        "Honda_HRV",
        "Mitsubishi_Xpander",
        "Suzuki_Swift",
        "plat",
    ]

    def fit(self, X, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict(self, X):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def score(self, X, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _output_prediction(self, prediction, confidence):
        return Classification(
            confidence=confidence, class_name=self.class_names[prediction]
        )
