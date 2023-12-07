import numpy as np
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Заголовок
st.subheader('Arvand Scoring')

# Загрузка моделей и данных
with open("CatBoost_model.pkl", "rb") as pickle_in:
    classifier1 = joblib.load(pickle_in)

with open("RF_model.pkl", "rb") as pickle_in:
    classifier2 = joblib.load(pickle_in)

with open("XGB_model.pkl", "rb") as pickle_in:
    classifier3 = joblib.load(pickle_in)

with open("days.pkl", "rb") as pickle_in:
    regression1 = joblib.load(pickle_in)

with open("sum.pkl", "rb") as pickle_in:
    regression2 = joblib.load(pickle_in)

# Создание DataFrame с данными
data1 = pd.DataFrame({'Nationality': ['Узбек', 'Точик', 'Тотор', 'Рус', 'Киргиз', 'Украин', 'Другие', 'Карис', 'Карачои']})
data2 = pd.DataFrame({'Filial': ['Истаравшан', 'Хучанд', 'Ч. Расулов', 'Душанбе', 'Исфара', 'Панчакент']})
data3 = pd.DataFrame({'Region': ['Шахристон', 'Гули сурх', 'Худжанд-Центр', 'Спитамен', 'Шарк', 'Мархамат', 'Душанбе', 'Навкент',
               'Кистакуз', 'Худжанд-Панчшанбе', 'Бустон', 'Истаравшан-филиал', 'Рудаки', 'Ашт', 'Калининобод',
               'Сино', 'Исфара', 'Хисор', 'Зафаробод', 'Ничони', 'Вахдат', 'Мехнатобод', 'Уяс', 'Дж.Расулов',
               'Конибодом', 'Дусти', 'Ниёзбек','Истаравшан', 'Рогун','Гончи', 'Чашмасор', 'Нофароч', 'Ободи',
               'Каракчикум', 'Оббурдон', 'Куруш', 'Ворух', 'Гулякандоз', 'Некфайз', 'Сомгор', 'Пунук', 'Панчакент',
               'Кулканд', 'Оппон', 'Файзобод', 'Турсунзода', 'Гусар', 'Равшан','Ифтихор', 'Х.Алиев', 'Ёри',
               'Мучун', 'Саразм']})
data4 = pd.DataFrame({'loan_goal': ['Животноводство и переработка молока', 'Приобретение техники', 'Ремонт дома', 'торговля',
                  'Земледелие', 'Приобретение мебели', 'Оплата на лечение', 'Проведение мероприятий', 'Оплата поездок',
                  'Услуги', 'Переоборудование транспорта', 'Потребнужды', 'Оплата образования', 'Производство',
                  'Покупка квартиры', 'Потреб.другое', 'Ремонт места деятельности', 'Сельское хозяйство', 'Все',
                  'Сушка фруктов', 'Коммерческий']})
data5 = pd.DataFrame({'sector': ['Животноводство', 'Потреб Экспресс', 'Потребнужды', 'Торговля', 'Зеленый кредит - Печки',
               'Сельхозкультура (ТАФФ)', 'Ремонт жилья', 'Бизнес Экспресс', 'Услуги', 'KFW - Ремонт жилья',
               'Сельхозтехника (ТАФФ)', 'KFW - Покупка и строит-во жилья', 'Производство', 'Образование',
               'Мигрант-бизнес 2', 'Мигрант-Потреб 2', 'Покупка и строит-во жилья', 'Товары в кредит', 'Корманд-кредит',
               'Мигрант', 'Старт-бизнес', 'Зеленый кредит - Солнечные батареи', ' Жилье для сотрудников', 'Достижения',
               'Сельхозкультура (Сароб)', 'Сельхозкультура-кредитная линия']})
data6 = pd.DataFrame({'pledge': ['Поручительство(Группа)', 'Недвижимость', 'Движимое имущество', 'Поручительство','Без залога']})
data7 = pd.DataFrame({'currency': ['Доллар США', 'Сомони', 'Рос.рубль']})
    
def issue_a_loan(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded):
    # Преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    # Преобразование всех переменных в числовой формат
    input_data = [
        Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
        Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
        high_debt
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), nationality_encoded, filial_encoded, region_encoded, 
                            loan_goal_encoded, sector_encoded, currency_encoded, pledge_encoded], axis=0)
                     
    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    prediction1 = classifier1.predict(input_array)
    prediction2 = classifier2.predict(input_array)
    prediction3 = classifier3.predict(input_array)
    total_pred = (prediction1 + prediction2 + prediction3) / 3
    total_pred = total_pred.astype(int)
    prediction2_1 = classifier1.predict_proba(input_array)
    prediction2_2 = classifier2.predict_proba(input_array)
    prediction2_3 = classifier3.predict_proba(input_array)
    total_pred2 = (prediction2_1 + prediction2_2 + prediction2_3) / 3
    total_pred2 = total_pred2.astype(int)
    return total_pred, total_pred2

def Delays_days(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
        Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
        high_debt
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), nationality_encoded, filial_encoded, region_encoded, 
                            loan_goal_encoded, sector_encoded, currency_encoded, pledge_encoded], axis=0)

    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    reg1 = regression1.predict(input_array)
    return reg1  

def Credit_sum(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded, issue):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
        Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
        high_debt, issue
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), nationality_encoded, filial_encoded, region_encoded, 
                            loan_goal_encoded, sector_encoded, currency_encoded, pledge_encoded], axis=0)

    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    reg2 = regression2.predict(input_array)    
    return reg2
                   
# Основная функция
def main():
    sex = st.radio("Укажите свой пол:", ['Мужской', 'Женский'])
    if sex == 'Мужской':
        Gender = 0
    else:
        Gender = 1
        
    # Создаем dummy-столбцы со значениями False для всех категорий
    nationality_encoded = pd.get_dummies(data1, prefix='', prefix_sep='').astype(bool)
    # Получаем выбор пользователя
    selected_nationality = st.selectbox('Введите национальность:', data1)
    # Если выбор сделан, обновляем соответствующий dummy-столбец в True
    if selected_nationality is not None:
        nationality_encoded = nationality_encoded[selected_nationality].astype(bool)
    
    Age = st.number_input('Сколько вам полных лет?', step=1, value=0)

    family_options = ['Оиладор', 'Беоила', 'Бевамард (бевазан)', 'Чудошуда']
    familyst = st.selectbox('Семейное положение:', family_options)
    label_encoder = LabelEncoder()
    encoded_family = label_encoder.fit_transform(family_options)
    FamilyStatus = encoded_family[family_options.index(familyst)]
    
    FamilySize = st.number_input('Сколько человек в семье?', step=1, value=0)
    
    educ =  ['Аспирантура', 'Оли', 'Миёнаи махсус', 'Олии нопурра', 'Миёна', 'Миёнаи нопурра']
    education = st.selectbox('Уровень образования:', educ)
    label_encoder = LabelEncoder()
    encoded_educ = label_encoder.fit_transform(educ)
    Education = encoded_educ[educ.index(education)]

    type = st.selectbox('Тип кредита:', ['Потребительский кредит','Кредит на предпринимательскую деятельность'])
    if type=='Потребительский кредит':
        type_of_credit = 0
    else:
        type_of_credit = 1
        
    filial_encoded = pd.get_dummies(data2, prefix='', prefix_sep='').astype(bool)
    selected_filial = st.selectbox('Филиал Банка:', data2)
    if selected_filial is not None:
        filial_encoded = filial_encoded[selected_filial].astype(bool)

    region_encoded = pd.get_dummies(data3, prefix='', prefix_sep='').astype(bool)
    selected_region = st.selectbox('Город\Регион проживания:', data3)
    if selected_region is not None:
        region_encoded = region_encoded[selected_region].astype(bool)

    sector_encoded = pd.get_dummies(data5, prefix='', prefix_sep='').astype(bool)
    selected_sector = st.selectbox('Сфера деятельности:', data5)
    if selected_sector is not None:
        sector_encoded = sector_encoded[selected_sector].astype(bool)

    loan_goal_encoded = pd.get_dummies(data4, prefix='', prefix_sep='').astype(bool)
    selected_goal = st.selectbox('Цель кредита:', data4)
    if selected_goal is not None:
        loan_goal_encoded = loan_goal_encoded[selected_goal].astype(bool)
        
    pledge_encoded = pd.get_dummies(data6, prefix='', prefix_sep='').astype(bool)
    selected_pledge = st.selectbox('Тип залога:', data6)
    if selected_pledge is not None:
        pledge_encoded = pledge_encoded[selected_pledge].astype(bool)
    
    currency_encoded = pd.get_dummies(data7, prefix='', prefix_sep='').astype(bool)
    selected_currancy = st.selectbox('Тип валюты:', data7)
    if selected_currancy is not None:      
        currency_encoded = currency_encoded[selected_currancy].astype(bool)

    Loan_amount = st.number_input('На какую сумму хотите взять кредит?', step=1, value=0) 

    Loan_term = st.number_input('На какой срок вы хотите взять кредит(месяц)?', step=1, value=0) 

    Lending_stage = st.number_input('Сколько кредитов вы брали(с учетом этого)?', step=1, value=0)
    if Lending_stage>1:
         have_delay = st.selectbox('Есть ли у вас просрочки?', ['Да','Нет'])
         if have_delay == 'Да':
             has_overdue = 1
         else:
             has_overdue = 0
    else:
        has_overdue = 0
    Repayment = st.number_input('Ежемесячная сумма погашения:', step=1, value=0)
    
    Grace_preiod = st.number_input('Льготный период (месяц):', step=1, value=0)
    
    have_debt = st.selectbox('У вас есть долги?', ['Да','Нет'])
    if have_debt == 'Да':
        Debt = st.number_input('Введите сумму долга:', step=1, value=0)
    else:
        Debt = 0
    if Debt > 10000:
        high_debt = 1
    else:
        high_debt = 0
        
    options = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-40', '40-50', '50+']
    bus_exp =  st.radio("Стаж работы:", options)
    label_encoder = LabelEncoder()
    encoded_busEx = label_encoder.fit_transform(options)
    business_experience = encoded_busEx[options.index(bus_exp)]

    Net_profit = st.number_input('Доход (в месяц):', step=1, value=0)
        
    result1 = ""
    result2 = ""
    result3 = ""
    result4 = ""
    if st.button("Predict"):
        result1, result2 = issue_a_loan(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded)
        if result1 == 0:
            result3 = Credit_sum(Gender, FamilySize, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded, result1)
            st.success(f'Сумму, которую вы ввели, пока что вам недоступно')
            st.success(f'Максимальная сумма, которую мы можем вам выдать: {result3}')
        else:
            st.success(f'Кредит будет выдан с вероятностью {result1[0]*100:.2f}%')
            st.success(f'Вероятность возврата кредита вовремя: {result2[0].item()*100:.2f}%')
            result4 = Delays_days(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded)
            st.success(f'Сколько примерно дней вы возможно просрочите: {(result4[0]).astype(int)}')
                                                   
if __name__ == '__main__':
    main()
