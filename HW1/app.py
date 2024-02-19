import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict
from charts import write_charts
from assessment import process_assesment
import time

def process_main_page():
    show_main_page()
    process_tabs()


def show_main_page():
    image = Image.open('data/piggy_bank.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Bank Clients",
        page_icon=image,

    )

    st.write(
        """
        # Классификация клиентов банка
        Один из способов повысить эффективность взаимодействия банка с клиентами — отправлять предложение о новой услуге не всем клиентам, 
        а только некоторым, которые выбираются по принципу наибольшей склонности к отклику на это предложение.
        Задача заключается в том, чтобы предложить алгоритм, который будет выдавать склонность клиента к положительному 
        или отрицательному отклику на предложение банка. Предполагается, что, получив такие оценки для некоторого множества клиентов, банк обратится с предложением только к тем, от кого ожидается положительный отклик.
        """
    )

    st.image(image)


def process_tabs():

    tab1, tab2, tab3 = st.tabs(["Исследовать", "Предсказать", "Оценить"])

    with tab1:
        write_charts()

    with tab2:
        process_inputs()

    with tab3:
        st.header('Оценка влияния признаков на отклик')
        st.markdown('**Что важно для отклика на маркетинговое предложение?**')
        most, least = process_assesment()
        st.dataframe(most)
        st.markdown('**Что не так важно для отклика на маркетинговое предложение?**')
        st.dataframe(least)



def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_inputs():
    st.header('Предсказание факта отклика')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]

    prediction, prediction_probas = load_model_and_predict(user_X_df)

    if st.button('Получить предсказание'):

        with st.spinner('Please wait...'):
            time.sleep(2)
        write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    sex = st.selectbox("Пол", ("Мужской", "Женский"))
    age = st.slider("Возраст", min_value=21, max_value=80, value=21,
                            step=1)
    work = st.radio("Статус занятости клиента", ("Работает", "Не работает"))
    pens = st.selectbox("Статус клиента относительно выхода на пенсию", ("Пенсионер", "Не пенсионер"))
    child = st.slider("Количество детей", min_value=0, max_value=20, value=0,
                            step=1)
    dependants = st.slider("Количество иждивенцев клиента", min_value=0, max_value=20, value=0,
                            step=1)
    income = st.number_input("Введите доход клиента (в рублях)")
    loan = st.number_input("Введите кол-во ссуд клиента (шт.)")
    closed_loan = st.number_input("Введите кол-во погашенных ссуд клиента (шт.)")

    translation = {
        "Мужской": 1,
        "Женский": 0,
        "Работает": 1,
        "Не работает": 0,
        "Пенсионер": 0,
        "Не пенсионер": 1
    }

    data = {
        'AGE': age,
        'SOCSTATUS_WORK_FL': translation[work],
        'SOCSTATUS_PENS_FL': translation[pens],
        'GENDER': translation[sex],
        'CHILD_TOTAL': child,
        'DEPENDANTS': dependants,
        'PERSONAL_INCOME': income,
        'LOAN_NUM_TOTAL': loan,
        'LOAN_NUM_CLOSED': closed_loan
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
