import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from os import path
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Penguin Classifier: A Machine Learning App')
st.write("This app uses 6 inputs to predict the species of penguin using "
         "a model built on the Palmer Penguins dataset. Use the form below"
         " to get started!")
file_path = path.abspath(__file__) 
dir_path = path.dirname(file_path)
password_guess = st.text_input('What is the Password?')
if password_guess != st.secrets['password']:
  st.stop()
penguins_file_path = st.file_uploader('Upload your own penguin data')
if penguins_file_path is None:
   penguins_file_path = path.join(dir_path,'penguins.csv')
model_file_path = path.join(dir_path,'random_forest_penguin.pickle')
output_file_path = path.join(dir_path,'output_penguin.pickle')
feature_importance_path = path.join(dir_path, 'feature_importance.png')
rf_pickle = open(model_file_path, 'rb')
map_pickle = open(output_file_path, 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()
with st.form('user_inputs'):
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    st.form_submit_button()
user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1
st.write(f"""the user inputs are {user_inputs}""".format())

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                               body_mass, island_biscoe, island_dream,
                               island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your penguin is of the {prediction_species} species")
st.write(
    """We used a machine learning (Random Forest)
    model to predict the species, the features
    used in this prediction are ranked by 
    relative importance below."""
)
st.image(feature_importance_path)
st.write(
    """Below are the histograms for each 
    continuous variable separated by penguin 
    species. The vertical line represents 
    your the inputted value."""
)
penguin_df = pd.read_csv(penguins_file_path)
penguin_df.dropna(inplace=True)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)



