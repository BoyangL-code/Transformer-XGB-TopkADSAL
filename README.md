A model for predicting aquatic animal effect toxicity concentrations.

Environment / Dependencies Requirements:
python 3.12, rdkit 2019.03.1.0, xgboost, pytorch, sklearn, optuna

Data Preparation：
After setting up the environment, you need to prepare the complete data file (.csv) and place it in the same folder as the Jupyter Notebook. The data file should meet the following format requirements:

Data Columns:

SMILES_Canonical_RDKit: SMILES string of the compound
mgperL: toxicity concentration
effect: effect type
Duration_Value: exposure duration (time)

If you need to change the column names, please modify them accordingly in the .ipynb file.

Model Training
Once the data file is ready, you can start training the model. Navigate to the folder containing the corresponding .ipynb file, then run the notebook step by step (execute the cells in order) to perform training.


