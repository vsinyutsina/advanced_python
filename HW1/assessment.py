import numpy as np
from pickle import load
import pandas as pd
import seaborn as sns

def process_assesment(path="models/log_reg.pkl"):

    cols = ['AGE', 'SOCSTATUS_WORK_FL',
            'SOCSTATUS_PENS_FL', 'GENDER', 'CHILD_TOTAL',
            'DEPENDANTS',
            'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']

    with open(path, "rb") as file:
        model = load(file)

    coefficients = model.coef_[0]

    cm_green = sns.color_palette("Greens", as_cmap=True)
    cm_red = sns.color_palette("Reds_r", as_cmap=True)

    feature_importance = pd.DataFrame({'Feature': cols, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    most, least = feature_importance.head(4), feature_importance.tail(4).sort_values('Importance')

    return most.style.background_gradient(cmap=cm_green), least.style.background_gradient(cmap=cm_red)

