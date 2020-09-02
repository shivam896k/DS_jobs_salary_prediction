# DS Jobs Salary Prediction

* Created a project to estimate data science job salary to help data scientiest negotiate their package.
* Extracted features from text data.
* Optimised Linear regression, Lasso regression and Random forest regressor using GridSearchCV to get the best parameters.


## Code and Resource Used

* python version: 3.8
* Packages: pandas, numpy, sklearn, matplotlib, seaborn, pickle


## Data Cleaning

* Extracted numeric data from salary
* Removed rows without salary
* Created new colunms for company state
* Made columns for employee provided salary and hourly wages
* Parsed rating column
* Made column for if location and company's headquaters are at same state
* Changed company founded date to age of company
* colunm for job title


## Model Building

* Converted categorical variables into dummy variables
* Split the data into train and test set of 8 : 2 ratio
* Tried three different models and evaluated their accuracy by "Mean Absolute Error"

* Models tried:
    * **Multiple Linear Regression**
    * **Lasso Resression**: Because of the sparse data, normalised regression like lasso would be effective
    * **Random Forest Regressor**: Due to sparsity in data, I thougth it would be good fit