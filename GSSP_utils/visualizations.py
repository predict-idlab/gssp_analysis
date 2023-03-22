__author__ = "Jeroen Van Der Donckt"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from typing import List


def plot_linear_classification_coefs(
    pipe: Pipeline, feat_cols: List[str], nb_to_show: int = 25
):
    """Plot the top (absolute) largest coefficients of the logistic regression model.
    The absolute coefficients are plotted on a horizontal barplot. The color indicates
    whether the coefficient is positive (green) or negative (red).

    Parameters
    ----------
    pipe: Pipeline
        The pipeline which contains the logistic regression model as last step.
    feat_cols: List[str]
        The column names of the features that are used in the pipeline.
    nb_to_show: int
        The number of feature coefficients to show, by default 25.
    """
    assert nb_to_show > 0
    model = (
        pipe[-1].best_estimator_ if hasattr(pipe[-1], "best_estimator_") else pipe[-1]
    )
    importances = model.coef_
    classes = pipe.classes_[1:] if len(pipe.classes_) == 2 else pipe.classes_
    for idx, label in enumerate(classes):
        sort_idx = np.argsort(np.abs(importances[idx]))[::-1]
        plt.figure(figsize=(10, 8))
        x = np.array(feat_cols)[sort_idx[:nb_to_show]][::-1]
        y = importances[idx][sort_idx[:nb_to_show]][::-1]
        color = ["lightgreen" if v > 0 else "salmon" for v in y]
        plt.barh(x, np.abs(y), color=color)
        plt.title(label)
        plt.show()