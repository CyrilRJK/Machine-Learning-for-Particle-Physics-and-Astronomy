# Machine-Learning-for-Particle-Physics-and-Astronomy-Exam

Element 1: the report is contained within MLPA_REPORT.pdf
Element 2: the code is contained within the following files
	- Generate_visualisations.ipynb: contains code to generate plots in report
	- Train_models.ipynb: contains code to train models as well as apply all pre-processing steps,
			put TrainingValidationData_200k_shuffle.csv in same folder as ipynb files first!
	- run_models.ipynb: contains code to load models, predict the test set and save predictions to csv,
			put ExamData.csv in same as ipynb files first.
	- processing.py file of utilty functions removed from notebooks for clarity.
Element 3: All models dicussed in the report are stored in the 'models' folder.
	   Load models with "tensorflow.keras.models.load_model()" function.
Element 4: Predictions are stored in predictions folder.
	   The multi-output model was designed for assignment (c) was used to make the binary predictions.
	   The multiclass LSTM model with application of the prior as discussed in section 3.4 of the report.

Code was written and cannot be guaranteed run without the latest versions of:
- Tensorflow
- Keras
- Seaborn 
- Pandas
			
