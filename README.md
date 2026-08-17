# Used Car Price Predictior
Machine learning project predicting UK used-car prices from vehicle characteristics including make, model, year, mileage, fuel type, transmission and engine size. Built in Python with pandas and scikit-learn, covering data cleaning, EDA, preprocessing, model comparison and Random Forest regression.

## Dataset

The project uses a dataset of approximately **100,000 used-car listings from the UK market**, containing vehicles from a range of manufacturers.

The dataset includes features such as:

* Make
* Model
* Year
* Transmission
* Mileage
* Fuel type
* Road tax
* MPG
* Engine size

The listed vehicle **price** is used as the target variable.

The data contains listings up to **2020**, meaning that the model represents the UK used-car market at that time and should not be interpreted as providing current market valuations.

**Dataset:** [100,000 UK Used Car Dataset - Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)

---

## Data Cleaning

The individual manufacturer datasets were combined into a single dataset and cleaned prior to analysis and modelling.

The cleaning process included:

* Standardising column names and data types across the individual datasets.
* Removing exact duplicate records.
* Checking categorical variables for inconsistent or unusual values.
* Investigating missing values.
* Investigating unrealistic vehicle years and other extreme values.
* Restricting the dataset to vehicles registered between **1996 and 2020**.
* Examining potential outliers in variables including price, mileage, MPG and road tax.

Some unusual price listings were retained where there was insufficient evidence to determine whether they were genuine high-value vehicles or erroneous listings.

---

## Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to understand the distributions and relationships within the dataset before modelling.

The analysis included:

* Distribution of vehicle prices.
* Distribution of vehicle mileage.
* Distribution of vehicle age/year.
* Distribution of MPG.
* Number of vehicles by manufacturer.
* Distribution of fuel types and transmission types.
* Relationships between numerical variables and vehicle price.
* Identification and investigation of potential outliers.

The analysis showed that vehicle prices and mileage were strongly right-skewed, while the dataset contained substantially more observations for some manufacturers than others.

---

## Model Application and Evaluation

The data was divided into separate **training and test sets**, with the test set kept separate during model development and hyperparameter optimisation.

Categorical variables were encoded using a preprocessing pipeline before being passed to the regression models.

Two tree-based machine-learning algorithms were compared:

* **Random Forest Regressor**
* **XGBoost Regressor**

Initial model performance was compared using **5-fold cross-validation** with Mean Absolute Error (MAE) as the primary evaluation metric.

The baseline Random Forest initially outperformed the baseline XGBoost model. However, XGBoost showed substantial improvement following hyperparameter optimisation.

Model performance was evaluated using:

* **Mean Absolute Error (MAE)**: the average absolute difference between predicted and actual prices.
* **Median Absolute Error**: the median size of the prediction errors, reducing the influence of extreme outliers.
* **Root Mean Squared Error (RMSE)**: an error metric that places greater weight on large prediction errors.
* **R² Score**: the proportion of variation in vehicle prices explained by the model.

Error analysis was also performed by comparing predicted and actual vehicle prices and investigating the largest prediction errors.

---

## Hyperparameter Optimisation

Hyperparameter optimisation was performed using **`RandomizedSearchCV`**.

The initial XGBoost model produced a 5-fold cross-validation MAE of approximately **£1,325**. After an initial hyperparameter search, this improved to approximately **£1,106**.

A second, more focused search around the best-performing parameter region further reduced the cross-validation MAE to approximately **£1,071**.

The best-performing XGBoost configuration was:

```python
{
    "subsample": 0.9,
    "n_estimators": 300,
    "min_child_weight": 2,
    "max_depth": 15,
    "learning_rate": 0.075,
    "colsample_bytree": 0.7
}
```

The final XGBoost model was selected for prediction based on its cross-validation performance.

---

## Limitations

* The dataset contains approximately **100,000 used cars from the UK market**, so the model may not generalise well to vehicles or specifications that are poorly represented in the training data.
* The dataset contains listings only up to **2020**. As a result, the predictor reflects used-car prices at that time and should not be interpreted as providing current market valuations.
* A few erroneous or unusual price listings remain in the dataset. Error analysis identified several extreme cases where the listed price appeared inconsistent with the make, model and vehicle characteristics.
* The available features do not capture all factors affecting a vehicle's value, such as trim level, optional equipment, condition and service history.

---

## Improvements and Extensions

* Perform more detailed **outlier detection and data cleaning**, particularly by examining price distributions within individual makes and models to identify erroneous listings.
* Engineer an **`age_in_years`** feature rather than relying solely on the vehicle's manufacturing year. This could make the model more adaptable to datasets from different time periods, although current market data would still be required to account for changes in used-car prices over time.
* Train the model on a **larger and more recent dataset** containing a wider range of makes, models and vehicle specifications to improve generalisation and enable predictions that better reflect the current used-car market.
* Extend the preprocessing pipeline to **handle missing values**, allowing users to obtain predictions when information such as MPG or road tax is unavailable.
* Incorporate additional features such as **trim level, vehicle condition and service history**, which may improve predictions, particularly for premium and high-value vehicles.

