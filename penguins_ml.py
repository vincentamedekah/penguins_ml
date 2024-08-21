import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from os import path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

file_path = path.abspath(__file__) 
dir_path = path.dirname(file_path) 
penguins_file_path = path.join(dir_path,'penguins.csv')
model_file_path = path.join(dir_path,'random_forest_penguin.pickle')
output_file_path = path.join(dir_path,'output_penguin.pickle')
feature_importance_path = path.join(dir_path, 'feature_importance.png')
penguin_df = pd.read_csv(penguins_file_path)
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
      'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))
rf_pickle = open(model_file_path, 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open(output_file_path, 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close() 
fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title('Which features are the most important for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig(feature_importance_path)