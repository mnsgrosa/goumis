import shap


class Explainer:
    def __init__(self, model, tokenizer):
        self.explainer = shap.Explainer(model, tokenizer)

    def explain(self, text):
        return self.explainer(text)
