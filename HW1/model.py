from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from pickle import load, dump
import pandas as pd


def split_data(df: pd.DataFrame, test=True):
    X = df.copy()

    if test:
        y = X['TARGET']
        X.drop('TARGET', inplace=True, axis=1)

        return X, y

    return X

def open_data(path="data/client_base.csv"):

    cols = ['TARGET', 'AGE', 'SOCSTATUS_WORK_FL',
            'SOCSTATUS_PENS_FL', 'GENDER', 'CHILD_TOTAL',
            'DEPENDANTS',
            'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']
    df = pd.read_csv(path)
    df = df[cols]

    return df


def preprocess_data(df: pd.DataFrame, test=True):

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = split_data(df, test=False)

    scaler = MinMaxScaler()
    X_df = scaler.fit_transform(X_df)

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="models/log_reg.pkl"):
    model = LogisticRegression()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="models/log_reg.pkl"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Клиент отвалится с вероятностью",
        1: "Клиент заинтересуется с вероятностью"
    }

    encode_prediction = {
        0: "Сожалеем, клиент не откликнется :(",
        1: "Ура! это заинтересованный клиент"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
