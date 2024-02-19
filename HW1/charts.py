import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def write_charts():

    st.header('Анализ клиенсткой аудитории банка')

    width = st.slider("plot width", 1, 25, 10)
    height = st.slider("plot height", 1, 25, 5)
    figsize = (width, height)

    df_min_data = pd.read_csv('HW1/data/client_base.csv')
    plt.figure(figsize=figsize)

    # charts

    with st.container():
        plot_age_target = sns.histplot(data=df_min_data,
                     x='AGE',
                     kde=True,
                     hue='TARGET')
        plot_age_target.set(title='Распределение возраста клиентов относительно факта отклика')
        st.pyplot(plot_age_target.get_figure(), clear_figure=True)
        st.write('''
        Если сравнивать клиентов относительно возраста и факта отклика, то видно, что тех, 
        кто откликнулся меньше, чем тех, кто проигнорировал. Так же нельзя сказать о том, 
        что откликаются преимущественно старики или молодеж - такой зависимости не наблюдается.
        ''')

    st.divider()

    with st.container():
        pens_status = df_min_data.groupby(['SOCSTATUS_PENS_FL', 'TARGET'])['AGREEMENT_RK'].count()
        pens_status = (
                    pens_status / df_min_data.groupby(['SOCSTATUS_PENS_FL']).AGREEMENT_RK.count() * 100).reset_index()
        pens_status['SOCSTATUS_PENS_FL'] = pens_status['SOCSTATUS_PENS_FL'].apply(
            lambda x: 'Пенсионер' if x else 'Не пенсионер')

        plot_pens = sns.barplot(data=pens_status,
                    hue='TARGET',
                    y='AGREEMENT_RK',
                    x='SOCSTATUS_PENS_FL',
                    palette=sns.color_palette("husl", 2))

        plot_pens.set(title='Распределение клиентов относительно пенсионного статуса', ylim=(0, 100))
        st.pyplot(plot_pens.get_figure(), clear_figure=True)
        st.write('''
        Чаще откликаются клиенты, социальный статус которых - не на пенсии.
        ''')

    st.divider()

    with st.container():

        df_child = df_min_data.copy()
        df_child['HAS_CHILD'] = df_child.CHILD_TOTAL > 0
        df_child_agg = df_child.groupby(['HAS_CHILD', 'TARGET'])['AGREEMENT_RK'].count()
        df_child_agg = (df_child_agg / df_child.groupby(['HAS_CHILD']).AGREEMENT_RK.count() * 100).reset_index()

        plot_child = sns.barplot(data=df_child_agg,
                    hue='TARGET',
                    y='AGREEMENT_RK',
                    x='HAS_CHILD',
                    palette=sns.color_palette("husl", 2))
        plot_child.set(title='Распределение клиентов относительно наличия детей', ylim=(0, 100))
        st.pyplot(plot_child.get_figure(), clear_figure=True)
        st.write('''
        Факт наличия детей видимо не влияет на факт отклика на маркетинговую кампанию.
        ''')

    st.divider()

    with st.container():

        social_status = df_min_data.groupby(['SOCSTATUS_WORK_FL', 'TARGET'])['AGREEMENT_RK'].count()
        social_status = (
                    social_status / df_min_data.groupby(['SOCSTATUS_WORK_FL']).AGREEMENT_RK.count() * 100).reset_index()
        social_status['SOCSTATUS_WORK_FL'] = social_status['SOCSTATUS_WORK_FL'].apply(
            lambda x: 'Работает' if x else 'Не работает')

        plot_work = sns.barplot(data=social_status,
                    hue='TARGET',
                    y='AGREEMENT_RK',
                    x='SOCSTATUS_WORK_FL',
                    palette=sns.color_palette("husl", 2))
        plot_work.set(title='Распределение клиентов относительно статуса работы')
        st.pyplot(plot_work.get_figure(), clear_figure=True)
        st.write('''
        Чаще откликаются клиенты, социальный статус которых - работают.
        ''')

    st.divider()

    with st.container():
        q999 = df_min_data.PERSONAL_INCOME.quantile(.999)

        plot_income = sns.violinplot(data=df_min_data[df_min_data.PERSONAL_INCOME < q999],
                    x='TARGET',
                    y='PERSONAL_INCOME',
                    hue='TARGET')
        plot_income.set(title='Распределение доходов клиентов относительно факта отклика')
        st.pyplot(plot_income.get_figure(), clear_figure=True)
        st.write('''
        У тех клиентов, которые откликаются, доход выше, чем у тех, кто не откликается. 
        Есть ли разница действительно - сказать сложно. 
        В таком случае имеет смысл посмотреть на наличие стат. значимой разницы.
        ''')

    st.divider()

    with st.container():
        plot_heatmap = sns.heatmap(data=df_min_data.corr(method='spearman'))
        plot_heatmap.set(title='Корреляция признаков с целевой переменной')

        st.pyplot(plot_heatmap.get_figure(), clear_figure=True)
        st.write('''
        Видно, что с целевой переменной больше всего коррелируют признаки - количество ссуд и 
        кол-во погашенных ссуд.
        ''')

