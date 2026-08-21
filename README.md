# Used Car Price Predictor

An end-to-end machine learning project that predicts historical UK used-car prices from vehicle characteristics and deploys the final model through an interactive **Streamlit web application**.

Built with **Python, pandas, scikit-learn and XGBoost**, the project covers the full workflow from raw data cleaning and exploratory analysis through model comparison, cross-validation, hyperparameter optimisation, error analysis and deployment.

[**Live Streamlit App**](YOUR_STREAMLIT_URL_HERE) | [**GitHub Repository**](YOUR_REPOSITORY_URL_HERE) | [**Dataset**](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)

> **Historical data note:** the model was trained on used-car listings up to 2020. Predictions therefore represent the market captured by the historical dataset and should not be interpreted as current used-car valuations.

---

## Project Highlights

| | Result |
|---|---|
| **Dataset** | ~100,000 UK used-car listings |
| **Final model** | Tuned XGBoost Regressor |
| **Final test MAE** | **£1,034** |
| **Median absolute error** | **£675** |
| **Median absolute percentage error** | **4.84%** |
| **Test R²** | **0.9657** |
| **Deployment** | Interactive Streamlit application |

**50% of test-set predictions had an absolute percentage error of 4.84% or less.**

---

## Business Problem and Project Objective

Used-car buyers can spend significant time comparing listings to understand whether a vehicle represents reasonable value.

This project investigates whether structured characteristics such as **make, model, age, mileage, engine size, fuel type and transmission** can be used to estimate a vehicle's listed price.

The machine learning model was then integrated into a Streamlit application to demonstrate how the analysis could be converted into a practical user-facing product.

With sufficiently recent training data, a system of this type could reduce the amount of manual price comparison required when assessing used-car listings.

---

## Live Application

The final model is deployed through an interactive **Streamlit application**.

[**Open the Streamlit App**](YOUR_STREAMLIT_URL_HERE)

Users can:

- Select a vehicle **make** from a dropdown.
- Select only **models associated with that manufacturer**.
- Enter numerical vehicle characteristics using sliders.
- Select transmission and fuel type.
- Generate an estimated vehicle price instantly.

The application loads the saved **complete scikit-learn pipeline**, meaning the same preprocessing used during model training is automatically applied to new user inputs before prediction.

The app is intended as a demonstration of deploying a machine learning model as a usable product. Because the underlying dataset ends in 2020, its predictions should not be treated as current market valuations.

---

## Project Workflow

The project followed the following workflow:

1. **Combined and cleaned** individual manufacturer datasets.
2. Performed **exploratory data analysis** and investigated unusual observations.
3. Created separate **training and test datasets**.
4. Built a reusable preprocessing pipeline using `ColumnTransformer` and `OneHotEncoder`.
5. Established **Linear Regression** as an initial benchmark.
6. Developed and compared **Random Forest and XGBoost** regression models.
7. Used **5-fold cross-validation** to compare model performance.
8. Optimised model hyperparameters using `RandomizedSearchCV`.
9. Performed a second focused XGBoost optimisation around the strongest parameter region.
10. Selected XGBoost as the final model based on cross-validation performance.
11. Evaluated the selected model on the held-out test data.
12. Performed detailed **prediction error and outlier analysis**.
13. Saved the complete fitted pipeline using `joblib`.
14. Deployed the model using **Streamlit**.

---

## Dataset

The project uses approximately **100,000 historical UK used-car listings** covering multiple manufacturers.

**Source:** [100,000 UK Used Car Dataset - Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)

### Features

| Feature | Description |
|---|---|
| `make` | Vehicle manufacturer |
| `model` | Vehicle model |
| `year` | Registration year |
| `transmission` | Transmission type |
| `mileage` | Recorded vehicle mileage |
| `fuelType` | Fuel type |
| `tax` | Road tax |
| `mpg` | Miles per gallon |
| `engineSize` | Engine size in litres |
| `price` | Listed price, used as the target variable |

The listings extend up to **2020**, which is an important limitation when interpreting model predictions.

---

## Data Cleaning

The individual manufacturer datasets were combined into a single modelling dataset before analysis.

Cleaning and validation included:

- Removing **1,475 exact duplicate records**.
- Standardising and checking categorical values.
- Investigating missing values.
- Investigating unusual transmission and fuel-type categories.
- Checking unrealistic registration years.
- Restricting vehicle years to **1996-2020**.
- Investigating extreme values in price, mileage, road tax and MPG.
- Reviewing unusually high and low vehicle listings.

Some unusual observations were retained where there was insufficient evidence to determine whether they represented genuine unusual vehicles or erroneous listings.

This avoided introducing subjective cleaning decisions solely to improve model performance.

---

## Exploratory Data Analysis

EDA was performed before modelling to understand the structure and quality of the dataset.

The analysis included:

- Vehicle price distribution.
- Mileage distribution.
- Registration year distribution.
- MPG distribution.
- Number of vehicles by manufacturer.
- Fuel-type and transmission distributions.
- Relationships between numerical features and vehicle price.
- Correlation analysis.
- Identification and investigation of potential outliers.

Vehicle prices and mileage were strongly **right-skewed**, while some manufacturers and models were much more heavily represented than others.

These findings informed both the modelling approach and later interpretation of prediction errors.

---

## Preprocessing Pipeline

The target variable was separated from the vehicle characteristics before creating the modelling datasets.

Categorical features were identified as:

```python
categorical_features = [
    "make",
    "model",
    "transmission",
    "fuelType"
]
```

A scikit-learn `ColumnTransformer` was used to:

- Apply **one-hot encoding** to categorical variables.
- Use `handle_unknown="ignore"` so previously unseen categories do not cause the pipeline to fail.
- Pass numerical variables through without transformation.

Preprocessing and modelling were combined using a scikit-learn `Pipeline`.

This ensured that exactly the same preprocessing was applied during:

- model training,
- cross-validation,
- test-set prediction,
- and prediction inside the Streamlit application.

---

## Modelling Approach

Three regression approaches were investigated:

### Linear Regression

Linear Regression was used as a simple initial benchmark.

Initial hold-out performance:

| Metric | Result |
|---|---:|
| MAE | £2,840 |
| RMSE | £4,682 |
| R² | 0.7766 |

The relatively high prediction error suggested that a simple linear relationship was unable to capture much of the complexity in vehicle pricing.

### Random Forest

Random Forest substantially improved predictive performance and initially appeared to be the strongest tree-based model.

Its **5-fold cross-validation MAE was £1,175.92**.

### XGBoost

The initial XGBoost configuration performed worse than Random Forest, with a **5-fold CV MAE of £1,325.44**.

However, XGBoost responded strongly to hyperparameter optimisation and ultimately became the best-performing model.

---

## Model Development Progression

Different evaluation stages served different purposes, so the results below should not be interpreted as a single like-for-like leaderboard.

| Model / Stage | Evaluation | MAE |
|---|---|---:|
| Linear Regression | Initial hold-out benchmark | £2,840 |
| Random Forest | 5-fold cross-validation | £1,176 |
| Baseline XGBoost | 5-fold cross-validation | £1,325 |
| XGBoost, first optimisation | 5-fold cross-validation | £1,106 |
| **XGBoost, final optimisation** | **5-fold cross-validation** | **£1,071** |
| **Final XGBoost** | **Held-out test data** | **£1,034** |

The baseline Random Forest initially outperformed XGBoost. Hyperparameter optimisation subsequently reduced XGBoost cross-validation MAE by approximately **19%**, resulting in the strongest final model.

---

## Hyperparameter Optimisation

Hyperparameter optimisation was performed using `RandomizedSearchCV`.

### First XGBoost Search

The initial search evaluated **20 parameter combinations using 5-fold cross-validation**, resulting in 100 model fits.

This reduced XGBoost CV MAE from:

**£1,325 → £1,106**

The strongest configuration placed `max_depth` at the upper limit of the original search range, suggesting that deeper trees were worth exploring.

### Focused Second Search

A second search narrowed the parameter ranges around the strongest-performing region while extending the search for tree depth.

- **30 candidate configurations**
- **5-fold cross-validation**
- **150 model fits**
- **6.8 minutes**
- Best CV MAE: **£1,071.30**

### Final Hyperparameters

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

A smaller Random Forest hyperparameter search was also conducted. It did not improve upon the baseline Random Forest configuration and was substantially more computationally expensive to evaluate.

XGBoost was therefore selected as the final model.

Further tuning was stopped after the focused second search rather than repeatedly optimising against the same cross-validation results.

---

## Final Model Performance

After model selection and optimisation, the final XGBoost pipeline was evaluated on the held-out test data.

| Metric | Final Test Result |
|---|---:|
| **Mean Absolute Error** | **£1,034.44** |
| **Root Mean Squared Error** | **£1,835.11** |
| **R²** | **0.9657** |
| **Median signed error** | **+£58.59** |
| **Median absolute error** | **£674.89** |
| **Median absolute percentage error** | **4.84%** |

The final model explains approximately **96.6% of the variation in listed vehicle prices** within the test dataset.

The median absolute error of approximately **£675** means that half of test-set predictions were within £675 of the listed price.

The **4.84% median absolute percentage error** means:

> 50% of test-set predictions had an absolute percentage error of **4.84% or less**.

The median signed error of only **+£58.59** suggests little directional bias in the typical prediction.

The final test MAE of **£1,034** was also close to the model's 5-fold CV MAE of **£1,071**, indicating that the cross-validation performance transferred well to unseen data.

---

## Error Analysis

Headline metrics alone do not show where a model succeeds or fails, so additional error analysis was performed.

This included:

- Examining the distribution of signed prediction errors.
- Comparing mean and median errors.
- Investigating the **20 largest absolute prediction errors**.
- Reviewing unusual vehicle listings.
- Comparing model performance across vehicle price bands.

### Error by Vehicle Price

| Vehicle Price | Vehicles | Median Absolute Error | Median Percentage Error |
|---|---:|---:|---:|
| < £10k | 4,992 | £436 | 5.60% |
| £10k-£20k | 9,400 | £661 | 4.69% |
| £20k-£40k | 4,500 | £1,169 | 4.47% |
| £40k+ | 538 | £2,400 | 4.96% |

Absolute errors naturally increased with vehicle price.

However, **relative error remained reasonably stable**, with median percentage error ranging from approximately **4.5% to 5.6%** across all price bands.

This suggests that the larger monetary errors observed for premium vehicles are substantially explained by their higher overall values rather than a dramatic decline in relative model performance.

The £40k+ category contains far fewer observations than the other groups, so its performance should be interpreted with greater caution.

### Largest Errors

Some of the largest errors occurred for listings with unusually high prices relative to their make, model and other recorded characteristics.

These were treated as **potential data-quality issues rather than automatically classified as errors**, because their validity could not be confirmed from the available dataset.

Other large errors occurred among premium and high-value vehicles. Important variables that were unavailable to the model, such as:

- trim level,
- optional equipment,
- vehicle condition,
- service history,
- performance variants,

may explain some of this remaining variation.

No test observations were retrospectively removed after inspecting the prediction errors, as doing so would artificially improve the reported performance.

---

## Potential Use Cases

The current application is based on historical data and is therefore a demonstration rather than a current-market valuation service.

With sufficiently recent and regularly updated training data, a similar system could potentially:

- **Estimate the market value of a listing**, helping buyers assess whether an asking price represents reasonable value.
- **Help buyers understand likely budgets** for particular makes, models and vehicle specifications.
- **Flag potentially underpriced or overpriced listings** by comparing asking prices with model estimates.
- **Support online vehicle marketplaces** by displaying estimated values alongside listings.
- **Assist dealerships with pricing decisions** when adding vehicles to inventory.
- **Support dealership purchasing decisions** by providing an initial data-driven estimate of potential resale value.

---

## Limitations

### Historical Market Data

The most important limitation is that the dataset contains listings only up to **2020**.

The model therefore learns the relationship between vehicle characteristics and prices within the historical market represented by that data.

It cannot account for subsequent changes in the UK used-car market and should **not be interpreted as providing reliable current valuations**.

### Dataset Coverage

The dataset contains approximately 100,000 vehicles, but representation is uneven across:

- manufacturers,
- models,
- specifications,
- price ranges.

Predictions may therefore be less reliable for poorly represented vehicles.

### Unusual Listings

Error analysis identified several observations with unusually high or low prices relative to their recorded characteristics.

Some may represent genuine unusual vehicles, while others may reflect data-quality issues. Their validity cannot be established from the available data alone.

### Missing Explanatory Variables

The dataset does not include several characteristics that can materially affect vehicle prices, including:

- trim level,
- optional equipment,
- condition,
- service history,
- accident history.

These omissions are particularly relevant when distinguishing between different specifications of premium vehicles.

---

## Future Improvements

### Use Newer and Multi-Year Market Data

The most important improvement would be to train the model using more recent listings.

A dataset covering multiple years would also allow the model to distinguish more clearly between:

- **vehicle depreciation**, and
- **changes in the overall used-car market**.

### Engineer Vehicle Age

Rather than relying solely on registration year, a future model could use:

```python
vehicle_age = listing_year - registration_year
```

With multi-year data, both `vehicle_age` and `listing_year` could potentially be included.

This would allow the model to learn vehicle ageing separately from broader changes in market conditions.

Simply applying general CPI inflation to 2020 predictions would not adequately solve the problem because used-car prices do not necessarily move in line with general consumer-price inflation.

### Improve Data Quality

Future work could include:

- more systematic outlier detection,
- price validation within individual make/model groups,
- investigation of suspicious listings before model training.

### Add More Vehicle Features

Useful additional variables could include:

- trim level,
- condition,
- service history,
- optional equipment,
- number of previous owners.

### Missing-Value Handling

The current application expects the required vehicle characteristics to be provided.

A future preprocessing pipeline could include explicit missing-value handling so users could still obtain predictions when information such as MPG or road tax is unavailable.

### Broader Application Validation

Further work could also include:

- more comprehensive input validation,
- limiting combinations to realistic vehicle specifications,
- additional automated testing,
- monitoring model performance as market data changes.

---

## Model Deployment

The final fitted pipeline was saved using `joblib`.

```python
joblib.dump(final_model, "../models/car_price_model.pkl")
```

The saved object contains both:

1. the fitted preprocessing pipeline, and
2. the tuned XGBoost model.

The Streamlit application therefore receives raw vehicle characteristics and applies the same transformations used during training before generating the prediction.

Valid make/model combinations were also exported from the modelling dataset and used by the application so that selecting a manufacturer dynamically filters the available model choices.

---

## Repository Structure

```text
used-car-price-predictor/
│
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
│
├── models/
│   ├── car_price_model.pkl     # Saved preprocessing + XGBoost pipeline
│   └── car_options.csv         # Valid make/model combinations
│
├── notebooks/                  # Data cleaning, EDA and model development
│
└── README.md
```

---

## Technologies

| Technology | Use |
|---|---|
| **Python** | Core analysis and modelling |
| **pandas** | Data cleaning and manipulation |
| **NumPy** | Numerical operations |
| **scikit-learn** | Pipelines, preprocessing, metrics, Random Forest and cross-validation |
| **XGBoost** | Final gradient-boosted regression model |
| **Matplotlib** | Exploratory analysis and error visualisation |
| **Streamlit** | Interactive web application |
| **joblib** | Model persistence |
| **GitHub** | Version control and project hosting |

---

## Running the App Locally

Clone the repository and navigate to the project folder.

Install the required packages:

```bash
pip install -r requirements.txt
```

Then launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Key Takeaways

This project demonstrates an end-to-end applied machine learning workflow:

- Cleaning and analysing a **large real-world dataset**.
- Building reusable **scikit-learn preprocessing pipelines**.
- Comparing multiple regression approaches using **cross-validation**.
- Improving model performance through structured **hyperparameter optimisation**.
- Evaluating performance using both headline metrics and detailed **error analysis**.
- Identifying and communicating limitations rather than removing inconvenient observations.
- Saving and deploying a complete fitted model pipeline.
- Translating a machine learning model into an interactive **user-facing Streamlit application**.

The final XGBoost model achieved a **£1,034 test MAE**, with **50% of test predictions within 4.84% of their listed price**, while the Streamlit application demonstrates how the modelling work can be converted into a practical analytical product.
