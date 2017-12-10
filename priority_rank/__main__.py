import pandas as pd
import numpy

from keras.models import Sequential
from keras.layers import Dense

from utils import PrimarySchoolDatasetHandler, WorkplaceDatasetHandler
from config import primaryschool, workplace

# Load Primary School dataset
# Read metadata
class_id, gender = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])

# Prepare csv for dataframe
PrimarySchoolDatasetHandler.prepare_training_dataset(
    primaryschool['dataset'], primaryschool['prepared_data'], gender)
primaryschool_df = pd.read_csv(primaryschool['prepared_data'], sep='\t')

# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata(workplace['metadata'])

# Prepare csv for dataframe
WorkplaceDatasetHandler.prepare_training_dataset(
    workplace['dataset'], workplace['prepared_data'], department)
workplace_df = pd.read_csv(workplace['prepared_data'], sep='\t')


# Transform network to training dataset
# Primary school
primaryschool_df['class1'] = primaryschool_df['class1'].astype('category')
primaryschool_df['gender1'] = primaryschool_df['gender1'].astype('category')
primaryschool_df['class2'] = primaryschool_df['class2'].astype('category')
primaryschool_df['gender2'] = primaryschool_df['gender2'].astype('category')

cat_columns = primaryschool_df.select_dtypes(['category']).columns
primaryschool_df[cat_columns] = primaryschool_df[cat_columns].apply(lambda x: x.cat.codes)

X = primaryschool_df.iloc[:, 0:2]
Y = primaryschool_df.iloc[:, 2:]


# Very very experimental model
numpy.random.seed(1)
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, epochs=10, batch_size=100)

scores = model.evaluate(X, Y)
print('{}: {}'.format(model.metrics_names[1], scores[1]))

# Priority Rank
"""
for each vertex n
    compute ranking R (m-lenght)
    for number of edges for vertex
        sample vertex t from the ranking
        add edge vertex - vertex t
"""
