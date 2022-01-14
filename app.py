import streamlit as st
import pandas as pd
pd.options.display.max_columns = None
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import warnings

header=st.container()
dataset=st.container()
features=st.container()
model_training=st.container()
model_results=st.container()

#@st.cache
#def get_data(filename):
    #telecom_data = pd.read_csv(filename)

    #return telecom_data
with header:
    st.header('WELCOME TO MY DATA SCIENCE PROJECT')
    st.write('In this proyect I look into the churn of a telco company')
    st.write('I make a model to predict whether a customer will leave the company or not')
    st.subheader('USING TREES!')
with dataset:
    st.header('1.Churn telco dataset')
    st.write("I've found this data set on: 'https://raw.githubusercontent.com/kaleko/CourseraML/master/ex2/data/ex2data1.txt'")
    telecom_data =pd.read_csv('telecom_churn copia.txt')
    st.write("This is my DATA before the EDA(Exploratory Data Analysis)")
    #st.text(telecom_data.head())
    df = pd.DataFrame(
        telecom_data)
    with st.expander("Show DATA"):
        st.dataframe(df)  # Same as st.write(df)
    st.write('"Churn" distribution chart on the Telco DataSet')
    total_amount_dist = pd.DataFrame(telecom_data['Churn'].value_counts()).head(50)
    with st.expander("Show Chart"):
        st.bar_chart(total_amount_dist)
    # Limpieza del DF
    # carga
    # cambiar nombre
    telecom_data.rename(columns={"Churn_num": "Churn"}, inplace=True)
    # se transforma a numerica; 1,0 en vez de yes, no
    telecom_data['Churn'] = telecom_data['Churn'].astype('int64')
    # dumificamos y creamos un df con las variables voice y ip (voice mail plan/international calls.
    vmp = pd.get_dummies(telecom_data['Voice mail plan'], drop_first=True, prefix="voice")
    ip = pd.get_dummies(telecom_data['International plan'], drop_first=True, prefix="ip")
    # eliminamos las columnas que había antes
    telecom_data.drop(['Voice mail plan', 'International plan'], axis=1, inplace=True)
    # unimos al df los dfs que hemos creado con las variables categóricas
    telecom_data = pd.concat([telecom_data, vmp, ip], axis=1)
    telecom_data.drop('State', axis=1, inplace=True)





with features:
    st.header("2.The features I've modified")
    st.markdown("* **Churn:** I convert it into numeric.")
    st.markdown("* **State:** I don't consider it an useful feature. I delete it.")
    st.markdown("* **Voice mail plan:** I convert it into numeric.")
    st.markdown("* **International plan:** I convert it into numeric.")
    st.write("This is my DATA after some EDA(Exploratory Data Analysis)")
    df = pd.DataFrame(
        telecom_data)
    with st.expander("Show DATA"):
        st.dataframe(df)  # Same as st.write(df)
    # st.text(telecom_data.head(5))
with model_training:
    st.header('3.Time to train the model')
    st.write('I use train_test_split to separate my data into "X_train", "X_test", "y_train", "y_test"')
    st.write('I use a Decision Tree Classifier for my model')
    #st.text('Choosee the hyperparameters of the model and see how the performance changes')
    sel_col, disp_col = st.columns(2)\

    #features= st.multiselect(
    #'What features should we use',
    #['Account length','Area code','Number vmail messages','Total day minutes','Total day calls',
     #'Total day charge','Total eve minutes','Total eve calls','Total eve charge','Total night minutes'
     #'Total night calls','Total night charge','Total intl minutes','Total intl calls','Total intl charge'
     #'Customer service calls','voice_Yes','ip_Yes',], help=("You can choose all! "))
    #st.write('You selected:', features)'''

    # MATRIZ DE ENTRENAMIENTO
    import random

    random.seed(113)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(telecom_data.drop('Churn', axis=1),
                                                        telecom_data['Churn'], test_size=0.25,
                                                        random_state=101)

    from sklearn import tree
    modelo = tree.DecisionTreeClassifier(max_depth=3)
    import warnings

    warnings.filterwarnings('ignore')

    modelo.fit(X_train, y_train)
with model_results:
    st.header("4. Let's see the predictions")
    st.write('Try my model with new DATA!')
    Account_length_new=st.slider('New Account_length',1,243,101)
    Area_code_new=st.slider('New Area_code',408,510,437)
    Number_vmail_messages_new=st.slider('New Number_vmail',0,51,8)
    Total_day_minutes_new=st.slider('New Total_day_minutes',0,350,179)
    Total_day_calls_new=st.slider('New Total_day_calls',0,165,100)
    Total_day_charge_new=st.slider('New Total_day_charge',0,59,30)
    Total_eve_minutes_new=st.slider('New Total_eve_minutes_new',0,363,200)
    Total_eve_calls_new=st.slider('New Total_eve_calls',0,170,100)
    Total_eve_charge_new=st.slider('New Total_eve_charge ',0,30,17)
    Total_night_minutes_new=st.slider('New Total_night_minutes',23,395,200)
    Total_night_calls_new=st.slider('New Total_night_calls ',33,175,100)
    Total_night_charge_new=st.slider('New Total_night_charge',1,17,9)
    Total_intl_minutes_new=st.slider('New Total_intl_minutes ',0,20,10)
    Total_intl_calls_new=st.slider('New Total_intl_calls',0,20,4)
    Total_intl_charge_new=st.slider('New Total intl charge',0,5,2)
    Customer_service_calls_new=st.slider('New Customer service calls',0,9,1)
    voice_Yes_new=st.selectbox('New voice_Yes. 1=Yes/0=No', options=[0,1], index=0)
    ip_Yes_new=st.selectbox('New ip_Yes1=Yes/0=No', options=[0,1], index=0)

    X_new = pd.DataFrame(
        {'Account length': [Account_length_new],'Area code': [Area_code_new],'Number vmail messages': [Number_vmail_messages_new],'Total day minutes': [Total_day_minutes_new],'Total day calls': [Total_day_calls_new],
     'Total day charge': [Total_day_charge_new],'Total eve minutes': [Total_eve_minutes_new],'Total eve calls': [Total_eve_calls_new],'Total eve charge': [Total_eve_charge_new],'Total night minutes': [Total_night_minutes_new],
     'Total night calls': [Total_night_calls_new],'Total night charge': [Total_night_charge_new],'Total intl minutes': [Total_intl_minutes_new],'Total intl calls': [Total_intl_calls_new],'Total intl charge': [Total_intl_charge_new],
     'Customer service calls': [Customer_service_calls_new],'voice_Yes': [voice_Yes_new],'ip_Yes': [ip_Yes_new]})

    st.write(X_new.head())
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    prediccion=modelo.predict(X_new)

    import graphviz

    # Export_graphviz
    cols = ['Account length', 'Area code', 'Number vmail messages', 'Total day minutes',
            'Total day calls', 'Total day charge', 'Total eve minutes',
            'Total eve calls', 'Total eve charge', 'Total night minutes',
            'Total night calls', 'Total night charge', 'Total intl minutes',
            'Total intl calls', 'Total intl charge', 'Customer service calls', 'voice_Yes', 'ip_Yes']
    dot_data = tree.export_graphviz(modelo,
                                    out_file=None,
                                    feature_names=cols)
    graph = graphviz.Source(dot_data)
    with st.expander("Show tree!"):
        st.graphviz_chart(dot_data)

    #if result:
        #st.graphviz_chart(dot_data)
    #result = st.button("Hide Tree")
    #st.write(result)
    #if result!=True:
        #st.graphviz_chart(dot_data)

    st.write(f"Predicction for the new DATA:{prediccion}\n")
    if prediccion == [0]:
        st.write("So, customer WILL NOT churn")
    else:
        st.write("So, customer WILL churn")

    st.subheader("How good is the prediction?")
    from sklearn.metrics import accuracy_score
    ac_train = round(accuracy_score(y_train, y_pred_train), 4)

    ac_test = round(accuracy_score(y_test, y_pred_test), 4)

    st.write(f"Precision on train set is **{ac_train}**")
    st.write(f"Precision on test set is **{ac_test}**")


    print('Degradation: ', round((ac_train-ac_test)/ac_train*100,2), '%')

    st.subheader("To sum up:")

    if prediccion == [0]:
        st.write(f"This customer **WILL NOT** churn with {(ac_test) * 100} % probability")
    else:
        st.write(f"This customer **WILL** churn with {(ac_test)*100:.2f} % probability")

    st.subheader("You can change the DATA and try again!")

with st.sidebar:
    st.header("✅THANKS FOR COMMING!")
    st.subheader("I hope you like my project")
    st.write("You can contact me on:⬇️")
    st.write("📲**LinkedIn:** 'https://www.linkedin.com/in/%C3%A1lvaro-casc%C3%B3n-puigdengolas-444b05183/'")
    st.write("💻**GitHub:** alvarocascon")
    st.write("📩**GMail:** alvarocascon@gmail.com")
