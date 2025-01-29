import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,precision_score, recall_score, f1_score,roc_auc_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder,StandardScaler
import numpy as np
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)
@st.cache_resource
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
@st.cache_data
def prepare_data(data):
    X = data.drop(columns=["y"])
    y = data["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    columns_to_encode = X_train.select_dtypes("object").columns
    encoder = TargetEncoder(random_state=42)
    X_train_encoded = encoder.fit_transform(X_train[columns_to_encode], y_train)
    X_test_encoded = encoder.transform(X_test[columns_to_encode])

    cols_to_standard = [col for col in X_train.columns if col not in columns_to_encode]
    standardscaler = StandardScaler()
    X_train_scaled = standardscaler.fit_transform(X_train[cols_to_standard])
    X_test_scaled = standardscaler.transform(X_test[cols_to_standard])

    X_train[cols_to_standard] = X_train_scaled
    X_test[cols_to_standard] = X_test_scaled
    X_train[columns_to_encode] = X_train_encoded
    X_test[columns_to_encode] = X_test_encoded

    return X_train, X_test, y_train, y_test, encoder, standardscaler, columns_to_encode, cols_to_standard
@st.cache_data
def get_proba(X_test): 
    y_pred_proba = model.predict_proba(X_test)[:,1]
    return y_pred_proba

data = load_data("data_clean.csv")
model = load_model("prod_model.pkl")
X_train, X_test, y_train, y_test, encoder, standardscaler, columns_to_encode, cols_to_standard = prepare_data(data)   
y_pred_proba = get_proba(X_test)
with st.sidebar:
    st.image(
        "https://zfunds-public.s3.ap-south-1.amazonaws.com/articlesImage/1625041532470")
    st.title("Aplikacja dla modelu przewidującego sukces kampanii")
    choice = st.radio("Panel nawigacyjny", ["EDA", "Model", "Predykcja"])
    st.info("W tej aplikacji znajdują się 3 sekcję: EDA, Model, Predykcja")

if choice=="EDA":
    st.title("Exploratory Data Analysis")
    @st.cache_resource
    def get_profile(data):
        return ProfileReport(data, title="Dataset Profile Report")
    profile = get_profile(data)
    st_profile_report(profile)

if choice=="Model":

    st.write("## Dynamiczny wybór progu decyzyjnego")
    threshold = st.slider("Wybierz próg decyzyjny (threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = round(accuracy_score(y_test, y_pred),4)
    precision = round(precision_score(y_test, y_pred),4)
    recall = round(recall_score(y_test, y_pred),4)
    f1 = round(f1_score(y_test, y_pred),4)
    roc_auc = round(roc_auc_score(y_test, y_pred_proba),4)

    
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', "Roc Auc"],
        'Score': [accuracy, precision, recall, f1, roc_auc]
    }
    metrics_df = pd.DataFrame(metrics)

    
    col1, col2 = st.columns([2, 1])  

    with col1:
        st.write(f"Typ modelu: XGBoost")
        st.write("Metryki modelu na zbiorze testowym:")
        st.dataframe(metrics_df)

    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))  
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_title('Macierz Pomyłek', fontsize=14, weight='bold') 
    ax.set_xlabel('Przewidywane wartości', fontsize=12)          
    ax.set_ylabel('Rzeczywiste wartości', fontsize=12)           
    ax.tick_params(axis='both', which='major', labelsize=10)     
    st.pyplot(fig)

    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label='Krzywa ROC', color='blue', linewidth=2)  
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Losowa klasyfikacja')  
    ax.set_title('Krzywa ROC', fontsize=14, weight='bold')         
    ax.set_xlabel('Odsetek fałszywie pozytywnych (FPR)', fontsize=12)  
    ax.set_ylabel('Odsetek prawdziwie pozytywnych (TPR)', fontsize=12)
    ax.legend(fontsize=10)                                          
    ax.grid(True, linestyle='--', alpha=0.6)                       
    st.pyplot(fig)

    feature_names = X_train.columns  

    
    importance_scores = model.feature_importances_

   
    def plot_feature_importance(importance_scores, feature_names):
        
        sorted_idx = np.argsort(importance_scores)
        
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_scores)), importance_scores[sorted_idx], align='center')
        plt.yticks(range(len(importance_scores)), np.array(feature_names)[sorted_idx])
        plt.xlabel('Ważność')
        st.pyplot(plt)  

   
    st.write("### Ważność cech dla modelu")
    plot_feature_importance(importance_scores, feature_names)

    
if choice=="Predykcja":

    st.write("## Sekcja Predykcji")

    
    st.write("### Wprowadź dane wejściowe:")

    categorical_features= {key: list(data[key].unique()) for key in columns_to_encode if data[key].nunique()>2}
    binary_features = {key: list(data[key].unique()) for key in data.columns if data[key].nunique()==2 and key not in categorical_features and key!="y"}
    continuous_features = {key: [data[key].min(), data[key].max()] for key in cols_to_standard if data[key].nunique()>2}

   
    categorical_inputs = {}
    for feature, options in categorical_features.items():
        categorical_inputs[feature] = st.selectbox(f"{feature}:", options)

    
    continuous_inputs = {}
    for feature, (min_val, max_val) in continuous_features.items():
        continuous_inputs[feature] = st.slider(f"{feature}:", min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2)

    
    binary_inputs = {}
    for feature, options in binary_features.items():
        binary_inputs[feature] = st.radio(f"{feature}:", options)

    input = {**categorical_inputs, **continuous_inputs, **binary_inputs}    

    inputs_df = pd.DataFrame([input])

    inputs_df = inputs_df[X_train.columns]
    
    st.write("### Wprowadzone dane:")
    st.dataframe(inputs_df)

    
    input_encoded = encoder.transform(inputs_df[columns_to_encode])
    input_scaled = standardscaler.transform(inputs_df[cols_to_standard])

    inputs_df[columns_to_encode] = input_encoded
    inputs_df[cols_to_standard] = input_scaled

    
    st.write("### Po transformacjach:")
    st.dataframe(inputs_df)
    
    input_proba = model.predict_proba(inputs_df)[:,1]
    
    st.write(f"### Prawdopodobieństwo przyjęcia oferty lokaty tego przypadku: {float(input_proba):.2f}")

