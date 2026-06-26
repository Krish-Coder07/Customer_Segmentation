# Telco Customer Churn Prediction and Customer Segmentation

This is a project I did on the Telco Customer Churn dataset. The original goal was just to predict churn using a Random Forest, but while working on feature importance I realised the same features (tenure, monthly charges, churn probability) could also be used to segment customers into meaningful groups using K-Means. So the project ended up doing both — a classification part and a clustering part.

## Dataset

Using the standard Telco Customer Churn dataset (`Telco_customer_churn.xlsx`) : 7043 customers, with the usual columns like tenure, contract type, monthly/total charges, internet & phone service add-ons, and a churn label.

Churn split in the data: 5174 stayed, 1869 left. So there's a class imbalance to keep in mind.

## What I did

### 1. EDA
Started with basic exploration before touching any model:
- Customers who churn have noticeably lower tenure than ones who stay (median tenure for churned customers is much lower, confirmed via boxplot).
- Monthly charges range from 18.25 to 118.75. People who churn tend to pay more on average (~74.4 vs ~61.2 for those who stay).
- Contract type turned out to be one of the strongest signals :  month-to-month customers churn way more (about 42.7% of them) compared to one-year (11.3%) and two-year (2.8%) contracts. Makes sense, less commitment = easier to leave.
- Checked correlation between tenure, monthly charges, churn value/score and CLTV. Tenure has a moderate negative correlation with churn (~ -0.35), which lines up with the boxplot observation.

### 2. Data Cleaning
- `Total Charges` was stored as an object/string instead of numeric (probably blank strings for new customers with 0 tenure), so converted it with `pd.to_numeric(errors='coerce')`. This created 11 NaNs, which I just filled with 0 since these were customers with 0 tenure anyway, i.e. they hadn't been charged yet.
- Dropped columns that don't add anything for segmentation/modeling — CustomerID, Count, Zip Code, Latitude, Longitude, Churn Label (redundant with Churn Value), Churn Score, CLTV, and City. Most of these are either IDs, geography, or values that are basically derived from churn itself (so keeping them would be leaking the target).

### 3. Encoding
One-hot encoded the categorical columns with `pd.get_dummies(drop_first=True)`. Ended up with 31 columns after encoding.

### 4. Train/Test Split
Standard 80/20 split using `train_test_split`, target being `Churn Value`.

## Churn Prediction — Random Forest

Went through a few iterations here instead of stopping at the first model:

**Baseline RF (100 estimators):**
- Accuracy: ~78.6%
- Recall (churn class): 0.51 — this was the actual problem. Accuracy looked fine but the model was missing almost half the customers who actually churn.

Since this is a churn use case, recall on the churn class matters a lot more than overall accuracy , missing a customer who's about to leave is worse than accuracy being a point or two lower. So I focused on pushing recall up from here.

**Balanced class weights:**
Tried `class_weight='balanced'` to deal with the imbalance — recall barely moved (0.52), accuracy went up slightly to 79.2%. Not the fix I was hoping for.

**Hyperparameter tuning (300 estimators, max_depth=10, balanced weights):**
Still basically the same recall (0.52). At this point it was clear that just tuning RF parameters wasn't going to solve it, the issue was probably feature noise rather than model capacity.

**Feature Selection:**
Looked at feature importances from the tuned model. Tenure, Total Charges, Contract (two year), Monthly Charges and Dependents came out as the top features. Two features — `Phone Service_Yes` and `Multiple Lines_No phone service` were essentially dead weight (these two are basically redundant/inverse of each other, doesn't add info), so I dropped them.

After removing these two features and retraining:
- Recall jumped from 0.52 to **0.74** — biggest improvement in the whole notebook.
- Precision on churn class dropped a bit (0.60) and overall accuracy dropped slightly to 78%, which is an acceptable trade-off given recall is the priority metric here.

**Cross-validation** (5-fold) on the final model:
- Mean CV accuracy: ~77.9%
- Mean CV recall: ~73.4% — consistent with the test set, so the improvement wasn't a fluke from the train/test split.

**ROC-AUC:** 0.857 — model separates churners from non-churners reasonably well.

## Customer Segmentation — K-Means

For the second half, I used the trained RF model to generate churn probabilities for each customer, then combined that with tenure, monthly charges and total charges to build a feature set for clustering (scaled with `StandardScaler` before clustering, since K-Means is distance-based).

Used the elbow method to pick K — WCSS dropped sharply till K=3 and flattened out after, so went with **3 clusters**.

Cluster profile (means):

| Cluster | Tenure | Monthly Charges | Total Charges | Churn Prob |
|---|---|---|---|---|# Telco Customer Churn Prediction and Customer Segmentation

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
| 0 | ~32 months | ~32.8 | ~1047 | 0.12 |
| 1 | ~11 months | ~72.0 | ~884 | 0.69 |
| 2 | ~58 months | ~90.4 | ~5278 | 0.23 |

Based on this I labeled them:
- **Cluster 0 — Budget Loyal Customers**: low charges, decent tenure, low churn risk.
- **Cluster 1 — High Risk New Customers**: short tenure, high charges, by far the highest churn probability. These are the ones a retention team should worry about first.
- **Cluster 2 — Loyal Premium Customers**: long tenure, highest charges, relatively low churn risk despite paying the most — probably happy with the service or just locked into long contracts.

## Conclusion

Putting both halves together — new customers on higher-priced plans (Cluster 1) are clearly the most likely to churn, and contract type plays a huge role in this. The practical takeaway is that the company should focus retention efforts on improving onboarding for new customers and possibly nudging month-to-month customers toward longer contracts, since that's where most of the churn risk is concentrated.

## Tech used
Python, pandas, numpy, matplotlib, seaborn, scikit-learn (RandomForestClassifier, KMeans, StandardScaler, train_test_split, cross_val_score)
