class BaseDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def preprocess(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def postprocess(self, results):
        raise NotImplementedError("This method should be overridden by subclasses")