# Essential Libraries for Developing

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import ml_model


# This function is used in Exploration Page for description of the dataset
def description():
    st.markdown("**battery_power** : Total energy a battery can store in one time measured in mAh")
    st.markdown("**blue** : Has bluetooth or not")
    st.markdown("**clock speed** : speed at which microprocessor executes instructions")
    st.markdown("**dual_sim** : Has dual sim support or not")
    st.markdown("**fc** : Front Camera mega pixels")
    st.markdown("**four_g** : Has 4G or not")
    st.markdown("**int_memory** : Internal Memory in Gigabytes")
    st.markdown("**m_dep** : Mobile Depth in cm")
    st.markdown("**mobile_wt** : Weight of mobile phone")
    st.markdown("**n_cores** : Number of cores of processor")
    st.markdown("**pc** : Primary Camera mega pixels")
    st.markdown("**px_height** : Pixel Resolution Height")
    st.markdown("**px_width** : Pixel Resolution Width")
    st.markdown("**ram** : Random Access Memory in Megabytes")
    st.markdown("**sc_h** : Screen Height of mobile in cm")
    st.markdown("**sc_w** : Screen Width  of mobile in cm")
    st.markdown("**talk_time** : longest time that a single battery charge will last when you are")
    st.markdown("**three_g** : Has 3G or not")
    st.markdown("**touch_screen** : Has touch screen or not")
    st.markdown("**wifi** : Has wifi or not")
    st.markdown("**price_range** : 0 to 3 means Low to High price range")
    
# This function is used for generating correlation heatmap
def correlation():
    corr_matrice = data.corr()
    fig,ax = plt.subplots(figsize=(14,14))
    sb.heatmap(corr_matrice, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=16)
    st.pyplot(fig)
    

# This sets the App Name and its Icon
st.set_page_config(page_title="Mobile Price Range Predictor", page_icon="🔍")
# Reading the dataset
data= pd.read_csv("train.csv")
# Show some dataset tuples in App
some = data.loc[:100]

# This is used for sidebar for navigation purpose
page = st.sidebar.selectbox('Select Page',['Dataset Exploration','Predictor','About Developer'])
# This is the content on Exploration Page
if page=='Dataset Exploration':
    st.title("Mobile Price Range Exploration")
    st.image("new1.jpg")
    st.write("This section helps in analysing the data, we explored a dataset containing various attributes of mobile phones, including battery_power, blue (Bluetooth availability), ram (Random Access Memory), and px_width (Pixel Resolution Width).. etc. The primary objective of this section was to understand the relationship between different features and the price range of the mobile phones. These insights are crucial for understanding the factors that contribute to the pricing of mobile phones and can guide manufacturers and consumers in making informed decisions.")
    st.markdown("**Dataset** - https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification")
    st.markdown("---")
    if st.checkbox("Show Data"):
        st.markdown("### *This is the some part of the Dataset*" )
        st.dataframe(some)
    st.markdown("---")
    if 'show_description' not in st.session_state:
        st.session_state.show_description = False

    # Toggle button to show/hide description
    if st.button("Description of Dataset"):
        st.session_state.show_description = not st.session_state.show_description

    # Show the dataset description based on the session state
    if st.session_state.show_description:
        description()
        
    st.markdown("---")
    
    if st.checkbox("Dataset Information"):
        st.markdown("### *This is the statistical information about dataset*")
        st.table(data.describe())
     
    st.markdown("---")
    #These graph shows the relationship with the target variable.
    graph = st.selectbox("Some Graphs",["Price Range VS Ram", "Price Range VS Pixel Width", "Price Range VS Internal Memory","Price Range VS Battery Power","Price Range VS Pixel Height","Price Range VS Screen Width"])
    if(graph=="Price Range VS Ram"):
        fig, ax = plt.subplots(figsize=(12,6))
        sb.barplot(x='price_range', y="ram", data= data, color="red", ax=ax)
        ax.set_title("Relation between Price and Ram")
        st.pyplot(fig)
    if(graph=="Price Range VS Pixel Width"):
        fig, ax = plt.subplots(figsize=(12,6))
        sb.barplot(x='price_range', y="px_width", data= data, color="red", ax=ax)
        ax.set_title("Relation between Price and Pixel Width")
        st.pyplot(fig)
    if(graph=="Price Range VS Internal Memory"):
        fig, ax = plt.subplots(figsize=(12,6))
        sb.barplot(x='price_range', y="int_memory", data= data, color="red", ax=ax)
        ax.set_title("Relation between Price and Internal Memory")
        st.pyplot(fig)
    if(graph=="Price Range VS Battery Power"):
        fig, ax = plt.subplots(figsize=(12,6))
        sb.barplot(x='price_range', y="battery_power", data= data, color="red", ax=ax)
        ax.set_title("Relation between Price and Battery Power")
        st.pyplot(fig)
    if(graph=="Price Range VS Pixel Height"):
        fig, ax = plt.subplots(figsize=(12,6))
        sb.barplot(x='price_range', y="px_height", data= data, color="red", ax=ax)
        ax.set_title("Relation between Price and Pixel Height")
        st.pyplot(fig)
    if(graph=="Price Range VS Screen Width"):
        fig, ax = plt.subplots(figsize=(12,6))
        sb.barplot(x='price_range', y="sc_w", data= data, color="red", ax=ax)
        ax.set_title("Relation between Price and Screen Width")
        st.pyplot(fig)
    st.markdown("---")
    
    
    if 'show_correlation' not in st.session_state:
        st.session_state.show_correlation = False

    # Toggle button to show/hide description
    if st.button("Inter-Correlation Heatmap"):
        st.session_state.show_correlation = not st.session_state.show_correlation

    # Show the dataset description based on the session state
    if st.session_state.show_correlation:
        correlation()
        st.markdown("<span style='background-color: red;'> This shows that **battery_power**, **int_memory**, **px_height**, **px_width**, **ram**, **sc_w** have high correlation with **Price_range**</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    



if page=='Predictor':
    st.title("Mobile Price Range Predictor")
    st.image("new1.jpg")
    st.write("In this section, we employed a machine learning model, Support Vector Machine(SVC) to predict the price range of mobile phones based on their attributes. After preprocessing the data and splitting it into training and testing sets, we trained the model using features such as battery_power, ram, px_width, and px_height, int_memory, sc_w. These predictions are crucial for consumers seeking to make informed purchasing decisions and for manufacturers aiming to optimize pricing strategies.")
    st.markdown("---")
    st.subheader("You need to select the values in order to predict the price range of mobile phones.")
    lr = ml_model.train_model(ml_model.preprocess_data(ml_model.load_data()))
    
    battery_power_input = st.slider("Battery Power", min_value=501, max_value=1998)
    int_memory_input = st.slider("Internal Memory", min_value=16, max_value=64)
    px_height_input = st.slider("Pixel Height", min_value=1, max_value=1960)
    px_width_input = st.slider("Pixel Width", min_value=500, max_value=1998)
    ram_input = st.slider("RAM", min_value=256, max_value=3998)
    sc_w_input = st.slider("Screen Width", min_value=1, max_value=18)
    st.markdown("---")
    
    user_input = pd.DataFrame({
        'battery_power': [battery_power_input],
        'int_memory': [int_memory_input],
        'px_height': [px_height_input],
        'px_width': [px_width_input],
        'ram': [ram_input],
        'sc_w': [sc_w_input]
    })
    
    
    ch,space,pr = st.columns(3)
    agree_checkbox = ch.checkbox("I Agree to the above information")
    st.markdown("---")
    if pr.button("Predict"):
        if agree_checkbox:
            prediction = ml_model.predict__(user_input, lr)[0]
            st.success(f"Predicted Price Range: {prediction}")
            accuracy = ml_model.evaluation(ml_model.preprocess_data(ml_model.load_data()))[0]
            st.info(f"Accuracy of the model trained : {accuracy*100}")
            cross = ml_model.evaluation(ml_model.preprocess_data(ml_model.load_data()))[1]
            st.info(f"Cross Validation Score of the model trained : {cross}")
            st.markdown("---")
            st.balloons()
        else:
            st.warning("Please agree to the information before predicting.")
            st.markdown("---")
  
    
if page=="About Developer":
    st.title("Developer")
    st.markdown("---")
    st.markdown("### About Me")
    st.markdown("Hello, I'm **Nishant Pratap Singh**, currently pursuing **B.Sc(H) Computer Science** from **Delhi University** and I'm a data enthusiast passionate about machine learning and data visualization.")
    st.markdown("---")
    # Contact information section
    st.markdown("### Contact Information")
    st.write("- Email: npratapsingh084@gmail.com")
    st.write("- LinkedIn: [Nishant Pratap Singh](https://www.linkedin.com/in/nishant-pratap-singh-b96871257/)")
    st.markdown("---")
    st.markdown("### Other Platforms")
    st.write("- Github : [Nishant Pratap Singh](https://github.com/Nishant-Pratap-Singh)")
    st.write("- Codechef : [Nishant Pratap Singh](https://www.codechef.com/users/nishant_0904)")
    st.markdown("---")
