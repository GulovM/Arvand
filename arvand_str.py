import numpy as np
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Заголовок
st.subheader('Arvand Scoring')

with open("lr_model2.pkl", "rb") as pickle_in:
    classifier = joblib.load(pickle_in)

with open("model_gb.pkl", "rb") as pickle_in:
    regression1 = joblib.load(pickle_in)

with open("model_gb2.pkl", "rb") as pickle_in:
    regression2 = joblib.load(pickle_in)

# Функции для предсказаний
def issue_a_loan(Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience):
    prediction = classifier.predict([[Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience]])
    prediction2 = classifier.predict_proba([[Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience]])[:, 1]
    return prediction, prediction2


def Delays_days(Loan_amount, Loan_term, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience, issue):
    reg1 = regression1.predict([[Loan_amount, Loan_term, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience, issue]])  
    return reg1  

def Credit_sum(Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience):
    reg2 = regression2.predict([[Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
                Net_profit, Age, Region_code, Direction_of_activity, business_experience]])  
    return reg2                  
                     
# Основная функция
def main():
    Age = st.number_input('Сколько вам полных лет?', step=1, value=0)

    region_options = ['Шахристон', 'Гули сурх', 'Худжанд-Центр', 'Спитамен', 'Шарк', 'Мархамат', 'Душанбе', 'Навкент', 'Кистакуз',
                  'Худжанд-Панчшанбе', 'Бустон', 'Истаравшан-филиал', 'Рудаки', 'Ашт', 'Калининобод', 'Сино', 'Исфара',
                  'Хисор', 'Зафаробод', 'Ничони', 'Вахдат', 'Мехнатобод', 'Уяс', 'Дж.Расулов', 'Конибодом', 'Дусти',
                  'Ниёзбек', 'Истаравшан', 'Рогун', 'Гончи', 'Чашмасор', 'Нофароч', 'Ободи', 'Каракчикум', 'Оббурдон',
                  'Куруш', 'Ворух', 'Гулякандоз', 'Некфайз', 'Сомгор', 'Пунук', 'Панчакент', 'Кулканд', 'Оппон', 'Файзобод',
                  'Турсунзода', 'Гусар', 'Равшан', 'Ифтихор', 'Х.Алиев', 'Ёри', 'Мучун', 'Саразм']

    selected_region = st.selectbox('Регион\город проживания:', region_options)
    label_encoder = LabelEncoder()
    encoded_region = label_encoder.fit_transform(region_options)
    Region_code = encoded_region[region_options.index(selected_region)]

    Loan_amount = st.number_input('На какую сумму хотите взять кредит?', step=1, value=0) 

    Loan_term = st.number_input('На какой срок вы хотите взять кредит?', step=1, value=0) 

    activity_options = ['Животноводство и переработка молока', 'Приобретение техники', 'Ремонт дома', 'Торговля',
                        'Земледелие', 'Приобретение мебели', 'Оплата на лечение',
                        'Проведение мероприятий', 'Оплата поездок', 'Услуги',
                        'Переоборудование транспорта', 'Потребнужды',
                        'Оплата образования', 'Производство', 'Покупка квартиры', 'Потреб.другое',
                        'Ремонт места деятельности', 'Животноводство', 'Сельское хозяйство', 'Все',
                        'Сушка фруктов', 'Коммерческий']
    selected_activity = st.selectbox('Цель кредита:', activity_options)
    label_encoder1 = LabelEncoder()
    encoded_activity = label_encoder1.fit_transform(activity_options)
    Direction_of_activity = encoded_activity[activity_options.index(selected_activity)]

    Lending_stage = st.number_input('Сколько кредитов вы брали(с учетом этого)?', step=1, value=0)
    Number_of_delays = st.number_input('Сколько раз вы просрочили срок кредита?', step=1, value=0)
    Days_of_delay = st.number_input('Сколько дней вы просрочили в послендий раз, когда брали кредит?', step=1, value=0)
    
    options = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-40', '40-50', '50+']
    style = """
    div[data-baseweb="radio"] {
    padding: 10px;
    margin: 10px;
    background-color: #f0f0f0;
    border-radius: 10px;
    }

    div[data-baseweb="radio"]:hover {
    background-color: #e0e0e0;
    }
    """
    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)
    business_experience =  st.radio("Стаж работы:", options)
    
    Gross_profit = st.number_input('Ваша валовая прибыль(разница между маржинальной прибылью и постоянными производственными расходами):', step=1, value=0)
    Net_profit = st.number_input('Чистая прибыль:', step=1, value=0)
        
    result1 = ""
    result2 = ""
    result3 = ""
    result4 = ""
    if st.button("Predict"):
        result1, result2 = int(issue_a_loan(Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
                    Net_profit, Age, Region_code, Direction_of_activity, business_experience)) 
        if result1 == 0:
            result3 = int(Credit_sum(Loan_amount, Loan_term, Days_of_delay, Number_of_delays, Lending_stage, Gross_profit, 
            Net_profit, Age, Region_code, Direction_of_activity, business_experience))
            st.success(f'Сумма кредита в случае отказа: {result3}')
        else:
            st.success('Кредит будет выдан.')
            st.success(f'Вероятность возврата кредита: {(result2*100).astype(int)}%')
            result4 = int(Delays_days(Loan_amount, Loan_term, Number_of_delays, Lending_stage, Gross_profit, 
                    Net_profit, Age, Region_code, Direction_of_activity, business_experience, result1))
            st.success(f'Сколько примерно дней вы возможно просрочите: {result4}')
                            
if __name__ == '__main__':
    main()
