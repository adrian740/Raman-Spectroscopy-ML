# Raman Spectroscopy ML
Using Random Forest and SVM to Process Raman Spectroscopy Data. The code in this repository inputs raman shift and counts for a range of x coordinates, determines using random forest and SVM the key raman shifts that correspond to a material, and can perform regression on an unkown sample to determine the makeup of that sample.

# Dependencies
This repository uses:
* Numpy
* Scipy
* Pandas
* Matplotlib

In addition, Orange3 was installed (via anaconda) to construct the model and evaluate data points.

# How to use the Code
The code is split into 3 python files and one Orange3 file. These are provided indivdualy for customizability reasons.

## Preprocessing (preprocessor.py)
Used to format the data into `.csv` files that will serve as the input to Orange3. Provide:
1. `<temp>`: the target temperature
2. `<n>`: The length of the vector of raman shifts.
3. (Optional) To train the model, you will need to specify one train dataset. To compile the train dataset, modify lines 29 and 31 as needed. 

This will need to be performed for all temperatures.

## Processing (processor.py)

## Orange3 (raman.ows)
Used to train and evaluate the model. Open the file using Orange3. Input the `<.csv>` files generated before. 'File' nodes need to be refreshed due to file size. To do this, double click on each 'File' node and press 'Refresh' or find the `<.csv>` file. Once completed, click on each 'Save Data' node to save the data files, eg `<res_120.csv>`.

## Post-Processing (result_processor.py)
