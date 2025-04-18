import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
with open('rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


def main():
    st.title("Congestion Prediction App")
    st.write("This app predicts congestion based on various features.")
    html_temp= """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Congestion Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.header("User Input Features")
    st.sidebar.markdown("Please enter the following details:")

    hour=st.text_input("Hour of the day (0-23)", "type here ")
    day=st.text_input("Day of the week (0-6)", "type here")
    month=st.text_input("Month of the year (1-12)", "type here")
    junction=st.text_input("Junction ID", "type here")
    weekday=st.text_input("Weekday (0-6)", "type here")

    result=""
    if st.button("Predict"):
        features = np.array([[hour, day, month, junction, weekday]])
        prediction = loaded_model.predict(features)
        result = f"Predicted Congestion Level: {prediction[0]}"
        st.success(result)
    if st.button("Visualize Data"):
        st.subheader("Data Visualization")
        data = pd.read_csv('data.csv')
        st.write(data.head())
        fig, ax = plt.subplots()
        sns.countplot(x='congestion_level', data=data, ax=ax)
        ax.set_title('Congestion Level Distribution')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
    