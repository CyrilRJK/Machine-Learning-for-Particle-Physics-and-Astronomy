import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def prepare_input(df, particle_count=False):
    
    """
    Function that collects multiple universal pre-processing steps that are applied to every model for the sake of
    code readability. The components of the pipeline are in order: log transform -> generate partcle counts(optional)
    -> categorically encode obj variable -> standardise all variables (except the obj) -> apply zero-padding.
    After these transformations the dataframe is split into the categorical and numerical features.
    :df: pandas DataFrame
    :particle_count: boolean
    """
    
    df[[f'E{x}' for x in range(1,11,1)]] = np.log(df[[f'E{x}' for x in range(1,11,1)]].astype(float))
    df[[f'pt{x}' for x in range(1,11,1)]] = np.log(df[[f'pt{x}' for x in range(1,11,1)]].astype(float))
    df['MET'] = np.log(df['MET'].astype(float))
    
    if particle_count:
        df['particlecount'] = [10-x/5 for x in df.isnull().sum(axis=1).tolist()]
        
    df = apply_cat_encoder(df)
    df = apply_standard_scaler(df)
    df = df.fillna(0, axis=1) # fill up nan values in the four vectors op te particles. Safe to use on the whole dataframe as the other columns do not have nan values. 

    cat_df = df[[f'obj{x}' for x in range(1,11,1)]+['process ID']]
    #df.drop([f'obj{x}' for x in range(1,11,1)], axis=1, inplace=True) 
    
    return df, cat_df


def read_data(PATH = '', FILENAME = 'TrainingValidationData_200k_shuffle.csv'):

	"""
	Read the dataframe while ensuring proper splitting of the 4vectors.
	:PATH: string
	"""		
	COLUMNS = ['event ID','process ID','event weight', 'MET', 'METphi'] + [f'par{x}' for x in range(1,11, 1)]
	COLS = [(f'obj{x}', f'E{x}', f'pt{x}', f'eta{x}', f'phi{x}') for x in range(1,11,1)]
	COLS = [item for sublist in COLS for item in sublist]

	PATH = PATH + FILENAME

	df = pd.read_csv(PATH, names=COLUMNS, sep=';')

	for x in range(1,11,1): 
		df[[f'obj{x}', f'E{x}', f'pt{x}', f'eta{x}', f'phi{x}']] = df[f'par{x}'].str.split(',', expand=True)
	df.drop([f'par{x}' for x in range(1,11, 1)], axis=1, inplace=True)

	return df

def transform_labels(label_column):
	"""
	Function to encode the process ID column into binary labels for foreground (4top) and background. Solely to be used for plotting purposes.
	:label_column: pandas series
	"""
	map = {'ttbarW': 'background', '4top': '4top', 'ttbarHiggs': 'background', 'ttbarZ': 'background', 'ttbar': 'background'}
	return [map[x] for x in label_column] 


def binary_encode_labels(label_column):
	"""
	Function to encode the process ID column into binary labels for foreground (4top) and background coded as integers to be used for model training.
	:label_column: pandas series
	"""
	map = {'ttbarW': 0, '4top': 1, 'ttbarHiggs': 0, 'ttbarZ': 0, 'ttbar': 0}
	return [map[x] for x in label_column]

def apply_standard_scaler(dataframe):
	"""
	Apply the SKlearn StandardScaler to all features except for the obj variables.
	:dataframe: Pandas DataFrame
	"""

	scaler = StandardScaler()

	columns_to_scale = ['MET', 'METphi', 'E1', 'pt1', 'eta1', 'phi1',  'E2', 'pt2', 'eta2', 'phi2', 
	    'E3', 'pt3', 'eta3', 'phi3', 'E4', 'pt4', 'eta4', 'phi4', 'E5', 'pt5', 'eta5', 'phi5', 'E6', 'pt6', 'eta6',
	    'phi6', 'E7', 'pt7', 'eta7', 'phi7',  'E8', 'pt8', 'eta8', 'phi8', 'E9', 'pt9', 'eta9', 'phi9', 'E10','pt10', 'eta10', 'phi10']

	features = dataframe[columns_to_scale]
	scaler.fit(features.values)
	features = scaler.transform(features.values)
	dataframe[columns_to_scale] = features

	return dataframe

def merge4Vectors(dataframe):

	"""
	Merge the 4vectors consisting of the E, pt, eta and phi variables into one list which is stored into 1 column of the dataframe.
	Subsequently remove the original columns.
	Example: 
	columns E1 with value 2, pt1:3.5, eta1:51.221, phi1:857.2 --> 4vector:[2, 3.5, 51.221, 857.2]
	:dataframe: Pandas DataFrame

	"""
	object_columns_indexes = [f'obj{x}' for x in range(1,11,1)]
	COLS = [(f'E{x}', f'pt{x}', f'eta{x}', f'phi{x}') for x in range(1,11,1)]
	COLS = [item for sublist in COLS for item in sublist]

	for col in object_columns_indexes:
	    source_col_loc = dataframe.columns.get_loc(f'{col}')
	    dataframe[f'4Vector{col[3:]}'] = dataframe.iloc[:,source_col_loc+1:source_col_loc+5].values.astype(float).tolist()
	    dataframe[f'4Vector{col[3:]}'] = [np.array(x) for x in dataframe[f'4Vector{col[3:]}']]

	dataframe.drop(COLS, axis=1, inplace=True)

	return dataframe


def multiclass_encode_labels(label_column):

	"""
	Function to encode the process labels as one-hot vectors
	:label_column: pandas series

	"""

	map = {'ttbarW': [0,0,0,1,0], '4top': [1,0,0,0,0], 'ttbarHiggs': [0,0,1,0,0], 'ttbarZ': [0,0,0,0,1], 'ttbar': [0,1,0,0,0]}

	return [map[x] for x in label_column]


def get_class_weights(label_column):

	"""
	taken from: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#train_a_model_with_class_weights
	Fetch class weights from the data, in this case the result will be a simple (1,4,4,4,4) distribution since each ttbar entry has 1/4th as many datapoints as the 4top events.
	Weights are encoded as a dictionary with the key being the index of that particular category in the one-hot encoded labels. e.g. the one-hot vector for a 4top event is 
	[1,0,0,0,0] thus the class weights for 4top events is encoded under the key 0. For explanation see:
	https://stackoverflow.com/questions/43481490/keras-class-weights-class-weight-for-one-hot-encoding/50695814

	:label_column: pandas series of string labels
	"""

	weights = {}
	counts = label_column.value_counts()
	weights[0] = (1/counts['4top'])*(len(label_column))/2.0
	weights[1] = (1/counts['ttbar'])*(len(label_column))/2.0
	weights[2] = (1/counts['ttbarHiggs'])*(len(label_column))/2.0
	weights[3] = (1/counts['ttbarW'])*(len(label_column))/2.0
	weights[4] = (1/counts['ttbarZ'])*(len(label_column))/2.0

	return weights


def get_prior(label_column):  
	"""
	Calculate a prior based on class probabilities.
	:label_column: pandas series of string labels
	"""
	
	counts = label_column.value_counts()
	probabilities = []

	probabilities.append(counts['4top']/sum(counts))
	probabilities.append(counts['ttbar']/sum(counts))
	probabilities.append(counts['ttbarHiggs']/sum(counts))
	probabilities.append(counts['ttbarW']/sum(counts))
	probabilities.append(counts['ttbarZ']/sum(counts))

	return np.array(probabilities)


def apply_cat_encoder(dataframe):

	"""
	Encode the object variable as a ordinal number. This representation is required for the categorical embeddings. 
	The numbers will not be used as features themselves.
	:dataframe: Pandas DataFrame
	"""


	map = {'b': 1, 'e+': 2, 'e-': 3, 'm+': 4, 'm-': 5, 'j': 6, 'g': 7, np.nan: 0}
	stepsize=len(dataframe)

	object_columns_indexes = [f'obj{x}' for x in range(1,11,1)]
	object_columns = pd.concat([dataframe[x] for x in object_columns_indexes])

	labelEnc = LabelEncoder()

	transformed_object_columns = [map[x] for x in object_columns]

	n = 0
	for i in object_columns_indexes:
		dataframe[i] = transformed_object_columns[n:n+stepsize]
		n += stepsize

	return dataframe