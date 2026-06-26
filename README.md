# Telcom Customer Churn Prediction and Customer Segmentation

This project predicts customer churn using Random Forest, and then uses those same results to segment customers into groups using K-Means. Started off as a simple churn classification task, but the feature importance results turned out to be useful for segmentation too, so both parts are included here.

## Dataset

Telco Customer Churn dataset (`Telco_customer_churn.xlsx`), 7043 customers in total. Columns include tenure, contract type, monthly and total charges, internet and phone service add ons, and a churn label.

Churn split in the data, 5174 stayed and 1869 left. So there is a class imbalance to keep in mind.

## EDA

Started with basic exploration before touching any model.

- Customers who churn have noticeably lower tenure than ones who stay, confirmed using a boxplot.
- Monthly charges range from 18.25 to 118.75. Customers who churn tend to pay more on average, around 74.4 compared to 61.2 for the ones who stay.
- Contract type turned out to be one of the strongest signals. Month to month customers churn the most, around 42.7%, while two year contract customers churn the least, around 2.8%. Makes sense, less commitment means it is easier to leave.
- Checked correlation between tenure, monthly charges, churn value, churn score and CLTV as well. Tenure has a moderate negative correlation with churn, around 0.35, which lines up with the boxplot trend.

## Data Cleaning

- `Total Charges` was stored as text instead of numbers, probably because some new customers had blank values since they had 0 tenure. Converted it using `pd.to_numeric`, which gave 11 NaN values. Filled those with 0 since those customers had not been billed yet.

## Dropping Irrelevant Columns

- Dropped columns that do not help with modeling or segmentation, things like CustomerID, Count, Zip Code, Latitude, Longitude, Churn Label, Churn Score, CLTV and City.
- Most of these are either ID columns, location columns, or values that are basically derived from churn itself, so keeping them would just leak the target.

## Encoding

- Categorical columns were one hot encoded using `pd.get_dummies`, which gave 31 columns after encoding.

## Train Test Split

- Did a regular 80/20 train test split with `Churn Value` as the target.

## Model Building (Random Forest)

- Baseline model gave 78.6% accuracy, but recall on the churn class was only 0.51.
- That basically means it was missing almost half the customers who actually churned, which is not great for this use case since recall matters more than accuracy here.

## Balanced Random Forest

- Tried `class_weight='balanced'` to handle the imbalance, but recall barely moved, only went up to 0.52.
- Accuracy went up slightly to 79.2% though.

## Hyperparameter Tuning

- Tuned the model further with 300 estimators and max depth of 10, along with balanced weights.
- Recall still stayed at 0.52, so tuning alone was not solving the actual problem.

## Feature Selection

- Looked at feature importance from the tuned model instead. Tenure, Total Charges, Contract (two year), Monthly Charges and Dependents came out as the top features.
- Two features, `Phone Service_Yes` and `Multiple Lines_No phone service`, were not adding much value since they are basically redundant with each other, so dropped both and retrained.
- Recall jumped from 0.52 to 0.74 after this, the biggest improvement in the whole project. Accuracy dropped slightly to 78%, which is a fair trade off given recall is the priority metric here.

## Cross Validation

- Ran 5 fold cross validation on the final model as well, and got similar numbers, recall around 73.4% and accuracy around 77.9%.
- So the improvement was consistent and not just a one off result from a lucky split.

## ROC Curve

- ROC AUC score came out to 0.857, which shows the model separates churners and non churners reasonably well.

## Customer Segmentation (K-Means)

- Used the churn probability from the trained Random Forest model, along with tenure, monthly charges and total charges, as features for clustering.
- Scaled all of these using `StandardScaler` first since K-Means is distance based and sensitive to scale.

## Elbow Method

- Used the elbow method to pick the number of clusters, and K=3 looked like the right choice since WCSS dropped sharply till that point and flattened out after.

## Fitting K-Means

| Cluster | Tenure | Monthly Charges | Total Charges | Churn Prob |
|---|---|---|---|---|
| 0 | ~32 months | ~32.8 | ~1047 | 0.12 |
| 1 | ~11 months | ~72.0 | ~884 | 0.69 |
| 2 | ~58 months | ~90.4 | ~5278 | 0.23 |

- Cluster 0 is Budget Loyal Customers, low charges and low churn risk.
- Cluster 1 is High Risk New Customers, short tenure with the highest churn probability by far, this is the group a retention team should focus on first.
- Cluster 2 is Loyal Premium Customers, long tenure and high charges, but still relatively low churn risk, probably because they are either satisfied with the service or locked into long contracts.

## Conclusion

- New customers on higher priced plans, which is Cluster 1, are clearly the most likely to churn, and contract type plays a big role overall.
- Practical takeaway is that the company should focus on improving the onboarding experience for new customers, and try to push month to month customers toward longer contracts since that is where most of the churn risk is coming from.

## Tech Used

Python, pandas, numpy, matplotlib, seaborn, scikit-learn, mainly RandomForestClassifier, KMeans, StandardScaler, train_test_split and cross_val_score.

## Future Work

- Feature selection mattered a lot more than hyperparameter tuning here, a good lesson for next time, check feature importance early instead of tuning blindly first.
- Could also try SMOTE or a different model like XGBoost later, to see if recall improves further without losing too much precision.
