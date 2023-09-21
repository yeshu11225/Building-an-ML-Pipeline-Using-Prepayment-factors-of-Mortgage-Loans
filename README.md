# Building-an-ML-Pipeline-Using-Prepayment-factors-of-Mortgage-Loans

## Problem Statement

To predict the mortgage backed securities prepayment risks.

Prepayment risk refers to the possibility that the borrowers will pay off their
mortgage loans earlier than expected, which can effect the cash flows of
mortgage-backed securities.

First look at the dataset!
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 291451 entries, 0 to 291450
Data columns (total 28 columns):
 #      Column                          Null_Count  Dtype       Unique_Count
---     ------                          ----------  -----       ------------
0       CreditScore                     0           int64       370
1       FirstPaymentDate                0           int64       66
2       FirstTimeHomebuyer              0           object      3
3       MaturityDate                    0           int64       96
4       MSA ?                           0           object      392
5       MIP                             0           int64       37
6       Units                           0           int64       5
7       Occupancy                       0           object      3
8       OCLTV                           0           int64       102
9       DTI                             0           int64       66
10      OrigUPB                         0           int64       375
11      LTV                             0           int64       97
12      OrigInterestRate                0           float64     254
13      Channel                         0           object      4
14      PPM                             0           object      3
15      ProductType                     0           object      1
16      PropertyState                   0           object      53
17      PropertyType                    0           object      7
18      PostalCode                      0           object      1767
19      LoanSeqNum                      0           object      291451
20      LoanPurpose                     0           object      3
21      OrigLoanTerm                    0           int64       61
22      NumBorrowers                    0           object      3
23      SellerName                      24994       object      21
24      ServicerName                    0           object      20
25      EverDelinquent                  0           int64       2
26      MonthsDelinquent                0           int64       174
27      MonthsInRepayment               0           int64       212
dtypes: int64(13), object(14), float64(1)
memory usage: 63755.0+ KB
```

The dataset has x records and 28 variables.

* `CreditScore` The standardized credit score used to evaluate the borrower during the loan origination process.
* `FirstPaymentDate` The month and year that the first scheduled payment on the load in due
* `FirstTimeHomebuyer` The indicator denoting whether a borrower on the loan qualifies as a first-time homebuyer.
* `MaturityDate` The month and year that the final scheduled payment on the load in due.
* `MSA` Mortgage Security Amount
* `MIP` Mortgage Insurance Percent, the percentage of mortgage insurance coverage obtained at origination in effect at the time the security was issued.
* `Units` The number of dwelling units in the mortgaged property at the time the loan was originated. 
* `Occupancy` The classification describing  the property occupancy status at the time the loan was originated.       
* `OCLTV` For reperforming, modified fixed-rate and modified step-rate loans, the ratio, expressed as a percentage,  obtained by dividing the amount of all known outstanding loans at origination by the value of the property.  
* `DTI` Debt-To-Income Ratio, The ratio obtained by dividing the total monthly debt expense by the total monthly income of the borrower at the time the loan was originated or modified.
* `OrigUPB`           
* `LTV` (Loan-to-Value) The ratio expressed as a percentage, obtained by dividing the amount of loan at origination by the value of the property.
* `OrigInterestRate` For reperforming, modified fixed-rate and modified step-rate loans, the interest rate of the loan as stated on the  note at the time the loan was originated.  For reperforming, modified fixed-rate and modified step-rate loans, the interest rate of the loan as stated on the  note at the time the loan was originated.  
* `Channel` The origination channel used by the party that delivered the loan
  to the issuer.
* `PPM` Prepayment Penalty Mortgage
* `ProductType` The type of real-state product.
* `PropertyState` The abbreviation of denoting the location of the property securing the loan.    
* `PropertyType` The classification describing the type of property that secures the loan.
* `PostalCode`        
* `LoanSeqNum` Loan Sequence Number        
* `LoanPurpose` The classification of the loan as either a  purchase money mortgage or a refinance mortgage at the time the loan was originated.        
* `OrigLoanTerm` For reperforming, modified fixed-rate and modified step-rate loans, the number of months in which regularly  scheduled borrower payments are due as stated on the note at the time the loan was originated.
* `NumBorrowers` The number of borrowers who, at the time the loan was originated, are obligated to repay the loan. 
* `SellerName` The name of the entity that sold the load to the issuer.
* `ServicerName` The name of the entity the services the loan during the current reporting period.      
* `EverDelinquent` (Assumption) An indication if burrower is delinquent.
* `MonthsDelinquent` Indicates the #months a borrower has been delinquent on their mortgage payments.
* `MonthsInRepayment` Represents the #months a borrower has been making regular mortgage payments.

> "Delinquent" is an adjective used to describe someone or something that
> fails to fulfill a duty or obligation, especially when it comes to financial or
> legal matters.

## Defining the dependent variable (Y)?

A higher value for `MonthsDelinquent` could suggest a lower likelihood of
prepayment. And a lower value for `MonthsInRepayment` might imply a higher
prepayment risk.


## NULL Placehodlers

Check the type of data that the nullable_columns (now nullified) should
actually hold in the dataset. If they should hold numerical data change their
d_type from 'O' to number.

```python
 nullified_columns = nullable_columns
nullified_columns 
['FirstTimeHomebuyer', 'MSA', 'PPM', 'PropertyType', 'PostalCode', 'NumBorrowers']
```

Turns out `MSA` and `NumBorrowers` actually hold numeric data type so we will
change these columns to numeric. Since `PostalCode` does not exactly undergo
any numeric operations we will keep it as string/object.
We will use `pd.DataFrame`'s `to_numeric()` method with `errors='coerce'` argument to convert `['MSA', NumBorrowers]` to numeric columns. Any errors will be coerced to `nan`.


## Missing Value Treatment

```python
Value counts in MSA:
--------------------
31084.0    9338
16974.0    8771
12060.0    6985
47644.0    6673
38060.0    6201
           ...
21940.0       3
49500.0       2
25020.0       1
10380.0       1
32420.0       1
Name: MSA, Length: 391, dtype: int64



Value counts in PostalCode:
---------------------------
94500    3776
30000    3637
85200    3280
48100    3246
48000    2988
         ...
34500       1
4700        1
33200       1
84200       1
41300       1
Name: PostalCode, Length: 891, dtype: int64

Value counts in NumBorrowers:
-----------------------------
2.0    187335
1.0    103777
Name: NumBorrowers, dtype: int64
```

Categorical / Discrete

```
Value counts in PropertyType:
-----------------------------
SF    244923
PU     27506
CO     18100
MH       723
LH       105
CP        72
Name: PropertyType, dtype: int64

Value counts in SellerName:
---------------------------
Ot    76943
CO    34479
FL    25573
FI    24581
ST    22243
NO    16184
OL     7776
PR     7365
BA     7093
GM     6566
BI     6407
G      4734
CH     4599
CR     4459
FT     4105
WA     3139
AC     3076
HO     2970
PN     2407
RE     1758

Value counts in FirstTimeHomebuyer:
-----------------------------------
N    184154
Y     29282
Name: FirstTimeHomebuyer, dtype: int64

Value counts in PPM:
--------------------
N    282125
Y      3921
Name: PPM, dtype: int64
```


Since `FirstTimeHomebuyer` and `PPM` have only 2 unique values and one appears
more often than the other, it's not appropriate to replace NULL values with the
mode. `PropertyType` has only 22 missing values and consists of 6 unique values. We will replace the NULL values in this column with the mode.

`PostalCodes` are normally associated with an area. So we will look at the
`PropertyState` column to impute missing postal code values. For each missing
`PostalCode` we will find the state and use the mode value of the `'PostalCodes'`
for only that area.

The "OCLTV" ratio, is a financial metric that measures the ratio of the total loan amount to the property's appraised value. It provides insight into the borrower's level of indebtedness and the risk associated with the mortgage.

## Inconsistent Value Treatment

1. `'G '` and `'GM'` probably points to the same entity in `'SellerName'`
1. There doesn't appear to be any inconsistency in the list of state
   abbreviations provided for the 50 U.S. states. Although there are 53 unique
   values. 3 of these point to U.S. territories rather than states.
1. In the column `PropertyType` we have `['CO' 'CP' 'LH' 'MH' 'PU' 'SF']`. But
   there is mention of `LH` in the variable definitions. The value counts in
   `PropertyType` column indicate there are only 105 (small compared to the
   total # of records) for `'LH'`. We will replace all 105 `LH` values with
   `MH`, because that is the closest match.
1. Trailing spaces in column `ProductType` and `ServicerName`. We will strip
   those.




## Drop all columns that have only one unique values.

1. Relevance - Columns with a single unique value offer no discriminatory power
   and do not contribute any meaningful information to your model. If the
   column is not relevant to your analysis or does not provide any insights, it
   may be reasonable to drop it.
1. Dimensionality Reduction: High dimensionality can lead to overfitting and
   slow down model training. Removing irrelevant or constant value columns can
   help mitigate these issues.
1. Computational Efficiency
1. Better interpretability
1. **Feature Engineering** - Sometimes, columns with constant values can be
   transformed or combined with other features to create new informative
   features.

Also, drop columns with as many unique values as the # of entires.






## Encoding

**Nominal variables** are categorical variables that represent distinct
categories with no inherent order or ranking among them. These categories are
usually represented using labels or names. In nominal variables, the categories
have no quantitative meaning, and you cannot compare or rank them in any
meaningful way. **Ordinal variables** are categorical variables that have
categories with a clear order or ranking. However, the differences between
categories are not necessarily equal or measurable. Ordinal variables represent
a relative ranking or hierarchy.

```python
In [64]: loans.categorical_features
Out[64]:
['FirstTimeHomebuyer',
 'Occupancy',
 'Channel',
 'PPM',
 'PropertyState',
 'PropertyType',
 'PostalCode',
 'LoanPurpose',
 'SellerName',
 'ServicerName']
```

* None of the variables listed appear to be inherently ordinal, as they do not exhibit a clear ranking or hierarchy among categories. Therefore all are nominal variables. So all of the categorical columns will be One-Hot Encoded.



Remaining Months to Maturity?

Although much columns seem to have no NULL values, some of these contained an
'X' value which indicated that these values were either "missing" or
"not-available". We will use the `match_placeholders` function to filter out
unique combinations of this placeholder and spaces and also identify which
columns have these placeholders.




`FirstPaymentDate` and `MaturityDate` held date values in the format `YYYYMM`.
So we will parse these into proper `np.datetime` values and calculate the
number of days between these dates to create a new feature `InvestmentPeriod`.

Let's take another look @ the dataset. We have fixed the d_type of some
columns.
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 291451 entries, 0 to 291450
Data columns (total 29 columns):
 #      Column                          Null_Count  Dtype       Unique_Count
---     ------                          ----------  -----       ------------
0       CreditScore                     0           int64       370
1       FirstPaymentDate                0           datetime    66
2       FirstTimeHomebuyer              78015       object      3
3       MaturityDate                    0           datetime    96
4       MSA                             39100       float64     392
5       MIP                             0           int64       37
6       Units                           0           int64       5
7       Occupancy                       0           object      3
8       OCLTV                           0           int64       102
9       DTI                             0           int64       66
10      OrigUPB                         0           int64       375
11      LTV                             0           int64       97
12      OrigInterestRate                0           float64     254
13      Channel                         0           object      4
14      PPM                             5405        object      3
15      ProductType                     0           object      1
16      PropertyState                   0           object      53
17      PropertyType                    22          object      7
18      PostalCode                      6           object      892
19      LoanSeqNum                      0           object      291451
20      LoanPurpose                     0           object      3
21      OrigLoanTerm                    0           int64       61
22      NumBorrowers                    339         float64     3
23      SellerName                      24994       object      21
24      ServicerName                    0           object      20
25      EverDelinquent                  0           int64       2
26      MonthsDelinquent                0           int64       174
27      MonthsInRepayment               0           int64       212
28      InvestmentPeriod                0           int64       194
dtypes: int64(12), datetime64[ns](2), object(12), float64(3)
memory usage: 66032.0+ KB
```


## Fixing Missinng Values

1. Seller Name 
    * d_type object
    * 


```python
# Parse Dates
loans['FirstPaymentDate'] = pd.to_datetime(loans['FirstPaymentDate'], format="%Y%m")
loans['MaturityDate'] = pd.to_datetime(loans['MaturityDate'], format="%Y%m")

# Create a new column w/ the # of days between `FirstPaymentDate` and
# `MaturityDate`.
loans['InvestmentPeriod'] = (loans['MaturityDate']
                             - loans['FirstPaymentDate']).dt.days 
```


## Modelling

Problem domain -> Binary Classification Problem

1. Data splitting
1. Initial model training
1. Model Performance Assessment  -> Use same assessment for both techniques.
   Since we are dealing with an imbalance dataset we will use both `accuracy`
   and `f1 score` for the reporting the final model performance.
1. Hyperparameter Tuning
1. Cross-validation
1. Select best hyperparameters
1. Final model training
1. Model evaluation on Test set.
1. Interpretability
1. Deployment?

> Steps 2 through 4 are iterative.

**Model Assessment**

* Understand what the model is doing.
    1. Confusion matrix
    1. Precision and Recall
    1. Area under the ROC  Curve (AUC)
        - Higher the *area under the ROC curve, the better the classifier*.
        - Min 0.5 AUC for tuned models. Aim for AUC > 0.5.

* Overall performance of the model (validation checks)
    1. Accuracy
    1. F1 score

### Regularized Logistic Regression

TODO: 

1. Understand logistic regression.
1. Understand overfitting and underfitting.
1. Understand how adding penalty terms (regularization) can overcome problems
   of base logistic regression.


We will use all:-
1. L1 `penalty = 'l1'` 
1. L2 `penalty = 'l2'` 
1. Elastic Net `penalty = 'elasticnet'`

> We will report final metrics for regularized logistic regression using
> elastic net.


First let's use plain logistic regression models and just add penalty terms.
Here's an example,

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load a sample dataset
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Binary classification: Setosa vs. non-Setosa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model with L1 regularization
model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

selected_features = [feature for feature, coef in zip(data.feature_names, model.coef_[0]) if coef != 0]
print("Selected Features:", selected_features) 
```

This will yield the most relevant features. Try both `liblinear` and `saga`
solvers for L1 and use `sag` for L2 regularization.


**Hyperparameter Tuning**

Use Bayesian search w/ CV for hyperparameter tuning.

```python
# Define the parameter search space for Bayesian optimization
param_space = {
    'C': (1e-6, 1e+6, 'log-uniform'),  # Inverse of regularization strength (similar to lambda)
    'l1_ratio': (0.0, 1.0),            # Elastic Net mixing parameter (balance between L1 and L2)
}

# Create a Logistic Regression model with Elastic Net regularization
model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42)

# Define the Bayesian optimization object
opt = BayesSearchCV(
    model,
    param_space,
    n_iter=50,  # Number of optimization steps (adjust as needed)
    cv=5,       # Number of cross-validation folds
    n_jobs=-1,  # Use all available CPU cores for parallel evaluation
    random_state=42,
)

# Perform Bayesian optimization to find the best hyperparameters
opt.fit(X, y)

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", opt.best_params_)
print("Best CV Score (Accuracy):", opt.best_score_)

# Optionally, train the model with the best hyperparameters on the entire dataset
best_model = opt.best_estimator_
best_model.fit(X, y
```

### Random Forest

TODO: Understand the backing algorithm.

**Most important hyperparameters to tune are:**

1. Number of decision trees i.e. `n_estimators` 
1. Number of random samples or the depth of the tree i.e. `max_depth`  
1. Size of the random subset of the features to consider at each splits. i.e.
   `max_features`

**Other hyperparameters to tune:**

1. `min_samples_split`, `min_samples_leaf`, `class_weight`.

> Use grid search algorithm for hyperparameter tuning. 
> `from sklearn.model_selection import GridSearchCV` GridSearchCV will perform
> hyperparameter tuning as well as CV.

**Feature Importance Plot**

Experiment with different sets of features to find the most informative ones.

1. Gini Importance: A higher Gini Importance score indicates that a feature is
   more important in making accurate predictions. Features that lead to
   significant reduction in Gini impurity when used in splits are considered
   more influential.






