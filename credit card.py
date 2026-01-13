#!/usr/bin/env python
# coding: utf-8

# # <b><u> Project Title : Predicting whether a customer will default on his/her credit card </u></b>

# # Predicting Credit Card Default Payments
# 
# ## Problem Description
# 
# In the realm of financial services, one of the most critical challenges is assessing and managing credit risk. Lenders, such as banks and credit card companies, must make informed decisions about extending credit to individuals while minimizing the risk of default. This problem description pertains to building a predictive model to assess the likelihood of credit card default for individual customers based on their historical financial behavior.
# 
# ### Background
# 
# Credit card default occurs when a credit cardholder fails to make the required minimum payment on their credit card account for a specified period. It has significant financial implications for both the cardholder and the issuing bank. Accurately predicting the probability of default can help banks proactively manage risk and make informed decisions.
# 
# ### Significant financial implications for both the cardholder and the issuing bank
# 
# ### For the Cardholder:
# 
# **Accumulation of Debt:** When a cardholder defaults on their credit card payment, the outstanding balance continues to accrue interest and late fees. This can quickly lead to a significant increase in the total debt owed.
# 
# **Negative Credit Score Impact:** Late or missed payments are reported to credit bureaus, resulting in a lower credit score for the cardholder. A lower credit score can make it difficult to obtain loans, mortgages, or other forms of credit in the future, and if they do get approved, it may come with higher interest rates.
# 
# **Higher Interest Rates:** If a cardholder defaults, the issuing bank may raise the interest rate on the card. This means that any new charges will accumulate interest at a higher rate, making it even harder to pay off the debt.
# 
# **Collection Efforts:** The issuing bank may employ collection agencies to recover the debt, which can result in frequent and sometimes aggressive collection calls, letters, or even legal actions.
# 
# **Legal Consequences:** In extreme cases, cardholders who default on their payments may face legal actions, such as lawsuits, wage garnishment, or liens on their assets.
# 
# **Difficulty in Obtaining New Credit:** Defaulting on a credit card can make it challenging to obtain new credit or financial products in the future, as lenders and creditors may be hesitant to extend credit to someone with a history of non-payment.
# 
# ### For the Issuing Bank:
# 
# **Loss of Revenue:** When cardholders default, the bank loses the expected interest and fees associated with the outstanding balance. This can lead to a significant revenue loss for the bank, especially if multiple cardholders default.
# 
# **Increased Risk Management Costs:** Banks must allocate resources to manage and mitigate credit card defaults. This includes hiring and training staff for collections, legal procedures, and credit risk assessment.
# 
# **Impact on Profitability:** Credit card default can directly impact a bank's profitability. Banks rely on the interest and fees charged to cardholders as a source of revenue. When defaults increase, it affects their bottom line.
# 
# **Reputation Damage:** Frequent or widespread defaults can harm the bank's reputation, making it less attractive to potential customers and investors.
# 
# **Regulatory Scrutiny:** Banks are subject to regulations governing credit card lending and collection practices. If they have a high default rate, it can attract regulatory scrutiny and potential fines.
# 
# **Risk Management Challenges:** High default rates can disrupt a bank's risk management strategies and require them to reassess their lending policies and practices.
# 
# ## Project Workflow
# 
# 1. **Data Collection:** Gather credit card payment data for customers, including historical payment behavior, demographic information, and other relevant features.
# 
# 
# 2. **Data Preprocessing:** Clean and prepare the data for analysis, including handling missing values, encoding categorical variables, and scaling features.
# 
# 
# 3. **Exploratory Data Analysis (EDA):** Explore the dataset to gain insights into the distribution of variables, relationships, and potential patterns related to credit card default.
# 
# 
# 4. **Feature Engineering:** Create or transform features that can improve the predictive power of the model.
# 
# 
# 5. **Model Building:** Train and evaluate machine learning models for predicting the probability of default. Common models include logistic regression, decision trees, random forests, and gradient boosting.
# 
# 
# 6. **Model Evaluation:** Use the metrics to assess the model's performance.
# 
# 
# 7. **Model Deployment:** If the model performs well, consider deploying it for real-time credit risk assessment.
# 
# 
# 8. **Documentation and Reporting:** Document the project's findings, including model performance, insights from EDA, and recommendations for risk management.
# 
# 
# **Note:** The success of this project depends on data quality, feature selection, and the choice of machine learning algorithms. Regular model monitoring and updates may be necessary to maintain its accuracy in a changing financial landscape.
# 

# # Data Description
# 
# ## Attribute Information
# 
# The dataset used for this research consists of a binary response variable, `default payment`, with values (Yes = 1, No = 0). The study utilizes the following 23 explanatory variables:
# 
# 1. **ID:** ID of each client.
# 
# 
# 
# 2. **LIMIT_BAL:** Amount of given credit (NT) dollars, including individual and family/supplementary credit.
# 
# 
# 3. **SEX:** Gender of the client (1 = male, 2 = female).
# 
# 
# 4. **EDUCATION:** Education level of the client (1 = graduate school, 2 = university, 3 = high school, 4 = others, 5 = unknown, 6 = unknown).
# 
# 
# 5. **MARRIAGE:** Marital status of the client (1 = married, 2 = single, 3 = others).
# 
# 
# 6. **AGE:** Age of the client in years.
# 
# 
# 7. **PAY_0:** Repayment status in September 2005 (-1 = pay duly, 1 = payment delay for one month, 2 = payment delay for two months, … 8 = payment delay for eight months, 9 = payment delay for nine months and above).
# 
# 
# 8. **PAY_2:** Repayment status in August 2005 (scale same as above).
# 
# 
# 9. **PAY_3:** Repayment status in July 2005 (scale same as above).
# 
# 
# 10. **PAY_4:** Repayment status in June 2005 (scale same as above).
# 
# 
# 11. **PAY_5:** Repayment status in May 2005 (scale same as above).
# 
# 
# 12. **PAY_6:** Repayment status in April 2005 (scale same as above).
# 
# 
# 13. **BILL_AMT1:** Amount of bill statement in September 2005 (NT dollar).
# 
# 
# 14. **BILL_AMT2:** Amount of bill statement in August 2005 (NT dollar).
# 
# 
# 15. **BILL_AMT3:** Amount of bill statement in July 2005 (NT dollar).
# 
# 
# 16. **BILL_AMT4:** Amount of bill statement in June 2005 (NT dollar).
# 
# 
# 17. **BILL_AMT5:** Amount of bill statement in May 2005 (NT dollar).
# 
# 
# 18. **BILL_AMT6:** Amount of bill statement in April 2005 (NT dollar).
# 
# 
# 19. **PAY_AMT1:** Amount of previous payment in September 2005 (NT dollar).
# 
# 
# 20. **PAY_AMT2:** Amount of previous payment in August 2005 (NT dollar).
# 
# 
# 21. **PAY_AMT3:** Amount of previous payment in July 2005 (NT dollar).
# 
# 
# 22. **PAY_AMT4:** Amount of previous payment in June 2005 (NT dollar).
# 
# 
# 23. **PAY_AMT5:** Amount of previous payment in May 2005 (NT dollar).
# 
# 
# 24. **PAY_AMT6:** Amount of previous payment in April 2005 (NT dollar).
# 
# 
# 25. **default.payment.next.month:** Default payment (1 = yes, 0 = no).
# 
# 
# This dataset provides valuable information for analyzing credit card default behavior and building predictive models to assess credit risk.
# 
# Please note that the data contains both numerical and categorical variables, and preprocessing may be required before applying machine learning algorithms.
# 

# ## **Objective:**

# 
# Objective of our project is to predict which customer might default in upcoming months. Before going any fudther let's have a quick look on defination of what actually meant by **Credit Card Default.**
# 
# 
# > We are all aware what is **credit card**. It is type of payment payment card in which charges are made against a line of credit instead of the account holder's cash deposits. When someone uses a credit card to make a purchase, that person's account accrues a balance that must be paid off each month.
# 
# 
# 
# 
# > **Credit card default** happens when you have become severely delinquent on your credit card payments.Missing credit card payments once or twice does not count as a default. A payment default occurs when you fail to pay the Minimum Amount Due on the credit card for a few consecutive months.
#  

# # Types of Data Analysis
# 
# ## Descriptive Analysis
# 
# Descriptive analysis is about summarizing and describing the main characteristics of a dataset. It provides an overview of what the data looks like, including its central tendencies, variability, and distribution. This type of analysis answers questions like "What happened?" or "What is the current state of affairs?" Common techniques include calculating mean, median, mode, standard deviation, and creating data visualizations such as histograms or bar charts.
# 
# ## Diagnostic Analysis
# 
# Diagnostic analysis goes a step beyond descriptive analysis by attempting to answer the question "Why did it happen?" It involves identifying the causes or factors contributing to a particular outcome or observation. Diagnostic analysis is often used when you want to understand the root causes of a problem or an event. Techniques may include regression analysis, hypothesis testing, and root cause analysis.
# 
# ## Predictive Analysis
# 
# Predictive analysis aims to answer the question "What is likely to happen in the future?" It involves building models based on historical data to make predictions or forecasts. Common techniques in predictive analysis include regression analysis, time series analysis, and machine learning algorithms. It's widely used for tasks like sales forecasting, stock price prediction, and customer churn prediction.
# 
# ## Prescriptive Analysis
# 
# Prescriptive analysis takes predictive analysis a step further by not only predicting future outcomes but also suggesting actions to optimize or improve those outcomes. It answers the question "What should we do about it?" Prescriptive analytics often involves the use of optimization techniques and decision support systems to recommend specific courses of action. It is used in areas like supply chain optimization, healthcare treatment recommendations, and personalized marketing campaigns.
# 

# In[1]:


# Importing all basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# importing all the libraries for feature engineering and model building
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from scipy.stats import randint


# In[3]:


#reading the data
file = 'UCI_Credit_Card.csv'
path = os.getcwd() + '\\'+ file
df = pd.read_csv(path, index_col='ID')
df.head()


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe().T


# # 1. Data Cleaning

# In[8]:


#Checking for null values
df.isnull().sum()


# In[9]:


#Checking for duplicates and dropping if any
df[df.duplicated(keep=False)]
df.duplicated().sum()
df.drop_duplicates(inplace=True)


# In[10]:


#checking for unique values
df.nunique()


# In[11]:


#defining value counts, create your own dataframe and pass through the function to get the value counts
def valueCounts(df):
    for col in df.columns:
        print(df[col].value_counts())


# In[12]:


df_SEM = df[["SEX", "EDUCATION", "MARRIAGE"]]
valueCounts(df_SEM)


# **Education**
# 
# > 1 = graduate school; 2 = university; 3 = high school; 4 = others

# As we can see in dataset we have values like 5,6,0 as well for which we are not having description so we can add up them in 4, which is Others.

# In[13]:


df['EDUCATION'] = df['EDUCATION'].replace([5, 6, 0], 4)


# **Marriage**
# 
# 
# > 1 = married; 2 = single; 3 = others

# We have few values for 0, which are not determined . So I am adding them in Others category.

# In[14]:


fil = df['MARRIAGE'] == 0
df.loc[fil, 'MARRIAGE'] = 3


# In[15]:


df_pay_status = df[['PAY_0', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
valueCounts(df_pay_status)


# In[16]:


df.rename(columns={'default.payment.next.month':'IsDefaulter'}, inplace=True)


# # 2. Exploratory Data Analysis

# ## **Dependent Variable:**

# In[17]:


plt.figure(figsize=(10,5))
sns.countplot(x = 'IsDefaulter', data = df)


# In[18]:


df['IsDefaulter'].value_counts(normalize=True)*100


# As we can see from above graph that both classes are not in proportion and we have imbalanced dataset.
# 
# 
# 

# ## **Independent Variable:**

# ### **Categorical  Features**

# We have few categorical features in our dataset. Let'Check how they are related with out target class.

# **SEX**
# 
# 
# 
# *   1 - Male
# *   2 - Female
# 

# In[19]:


df['SEX'].value_counts(normalize=True)


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a crosstab to get the counts of 'IsDefaulter' for each 'SEX' category
gender_crosstab = pd.crosstab(df['SEX'], df['IsDefaulter'])

# Set a color palette
colors = ["#007acc", "#ff7f0e"]

# Create a stacked bar plot
sns.set(style="whitegrid")
ax = gender_crosstab.plot(kind='bar', stacked=True, color=colors)

# Customize the plot
ax.set_title('Defaulters by Gender', fontsize=16)
ax.set_xlabel('Sex', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.xticks(rotation=0)  # Rotate x-axis labels if needed

# Customize the legend and move it outside
legend_labels = ['Not Default', 'Default']
legend = plt.legend(legend_labels, title='Is Defaulter', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid lines
ax.yaxis.grid(which="both", color='gray', linestyle='--', linewidth=0.5)

plt.show()


# In[21]:


gender_crosstab = pd.crosstab(df['SEX'], df['IsDefaulter'], normalize='index')
gender_crosstab


# **Plotting our categorical features**

# In[22]:


categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
df_cat = df[categorical_features]
df_cat['Defaulter'] = df['IsDefaulter']
df_cat


# In[ ]:





# In[23]:


# df_cat=df_cat[['EDUCATION','MARRIAGE','SEX','Defaulter']].astype('str')


# In[24]:


df_cat.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)


# In[25]:


df_cat


# In[26]:


sns.countplot(data = df_cat, x = 'SEX', hue = 'MARRIAGE')


# In[27]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df_cat is your DataFrame

# # Define a bright color palette
# bright_palette = sns.color_palette("paired")

# # Create the countplot with the specified colors
# sns.set(style="whitegrid")  # Optional: Set the background style
# sns.countplot(data=df_cat, x='SEX', hue='MARRIAGE', palette=bright_palette)

# # Customize the plot further if needed
# plt.title('Countplot of SEX with MARRIAGE', fontsize=16)
# plt.xlabel('SEX', fontsize=14)
# plt.ylabel('Count', fontsize=14)
# plt.legend(title='MARRIAGE', title_fontsize='12')

# # Show the plot
# plt.show()


# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the color palette for the countplot
sns.set_palette("Set1")

# Create a figure with subplots
plt.figure(figsize=(15, 6))
grid = plt.GridSpec(1, 3, wspace=0.4)

for col in categorical_features:
    plt.figure(figsize=(10,5))
    fig, axes = plt.subplots(ncols=2,figsize=(13,8))
    df_cat[col].value_counts().plot(kind="pie",ax = axes[0],subplots=True,autopct='%1.1f%%')
    sns.countplot(x = col, hue = 'Defaulter', data = df_cat,)




# Below are few observations for categorical features:
# 
# 
# 
# 
# *   There are more females credit card holder,so no. of defaulter have high proportion of females.
# *   No. of defaulters have a higher proportion of educated people  (graduate school and university)
# *  No. of defaulters have a higher proportion of Singles.
# 
# 
# 
# 
# 

# In[29]:


df_cat["SEX"].value_counts(normalize=True)


# In[30]:


pd.crosstab(df_cat["SEX"], df_cat["Defaulter"], normalize="index")


# In[31]:


df_cat.groupby(['Defaulter', 'EDUCATION'],as_index=False)['EDUCATION'].count()


# In[32]:


df_cat[['Defaulter', 'EDUCATION']].value_counts()


# 
# ### Also you can do categorical analysis for other variables like marriage and education with respect to target column as shown above.

# ## Numerical Features
# 
# 
# 
# 
# 
# 
# **Limit Balance**

# In[33]:


df['LIMIT_BAL'].max()


# In[34]:


df['LIMIT_BAL'].min()


# In[35]:


df['LIMIT_BAL'].describe()


# In[36]:


df[(df['LIMIT_BAL'] >= 600000) & (df['IsDefaulter'] == 1)]


# In[37]:


sns.histplot(df['LIMIT_BAL'], kde = True)


# In[38]:


df['LIMIT_BAL'].skew()


# In[39]:


sns.barplot(x='IsDefaulter', y='LIMIT_BAL', data=df)


# In[40]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(x="IsDefaulter", y="LIMIT_BAL", data=df)


# In[41]:


sns.displot(data=df, x = 'LIMIT_BAL', hue = 'IsDefaulter', kind='kde')


# ##### The distribution is right skewed and the above plot explains that limit balance is a good feature to explain our targeT feature as the distribution is no overlapping.

# In[42]:


df.groupby(['IsDefaulter'])['AGE'].mean()


# In[43]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(x="IsDefaulter", y="AGE", data=df)


# In[44]:


#renaming columns 
df.rename(columns={'PAY_0':'PAY_SEPT','PAY_2':'PAY_AUG','PAY_3':'PAY_JUL','PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR'},inplace=True)
df.rename(columns={'BILL_AMT1':'BILL_AMT_SEPT','BILL_AMT2':'BILL_AMT_AUG','BILL_AMT3':'BILL_AMT_JUL','BILL_AMT4':'BILL_AMT_JUN','BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR'}, inplace = True)
df.rename(columns={'PAY_AMT1':'PAY_AMT_SEPT','PAY_AMT2':'PAY_AMT_AUG','PAY_AMT3':'PAY_AMT_JUL','PAY_AMT4':'PAY_AMT_JUN','PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'},inplace=True)


# **Bill Amount**

# In[45]:


bill_amnt_df = df[['BILL_AMT_SEPT','BILL_AMT_AUG','BILL_AMT_JUL','BILL_AMT_JUN','BILL_AMT_MAY','BILL_AMT_APR']]


# In[46]:


bill_amnt_df


# In[47]:


sns.histplot(df['BILL_AMT_SEPT'], kde= True)


# In[48]:


sns.pairplot(data = bill_amnt_df)


# In[49]:


bill_amnt_df.corr()


# **History payment status**

# In[50]:


pay_col = ['PAY_SEPT','PAY_AUG','PAY_JUL','PAY_JUN','PAY_MAY','PAY_APR']
for col in pay_col:
  plt.figure(figsize=(10,5))
  sns.countplot(x = col, hue = 'IsDefaulter', data = df)


# ### similarly you can do analysis on numerical features w.r.t. target variables

# In[51]:


X = df.drop(['IsDefaulter'],axis=1)
y = df['IsDefaulter']


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[53]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # 3. Feature Engineering

# ## Since our dataset is imbalanced we go for SMOTE TOMEK technique

# In[54]:


from imblearn.combine import SMOTETomek
smote = SMOTETomek()

# fit predictor and target variable
X_smote, y_smote = smote.fit_resample(X_train, y_train )

print('Original dataset shape', len(X_train))
print('Resampled dataset shape', len(y_smote))


# In[ ]:





# In[55]:


#checking wether the datsets are balanced
smote_df = X_smote.copy()
smote_df['IsDefaulter'] = y_smote
sns.countplot(x='IsDefaulter', data = smote_df)


# In[56]:


sns.displot(data=smote_df, x = 'LIMIT_BAL', hue = 'IsDefaulter', kind='kde')


# In[57]:


X_smote.shape, y_smote.shape


# ### Creating  a pipeline for preprocessing steps like encoding and scaling

# In[58]:


#1. It should create marital status - (married male, married female, single male.....)
#2. drop both the columns gender, marriage
#3. pay scales drop certain features and combine certain features 
#4. add column called dues (bill - payment)
def create_custom_pipeline(categorical_cols, numerical_cols, scaling_method=None):
    # Define transformers for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False, drop="first", handle_unknown='ignore'))])

    if scaling_method:
        if scaling_method == 'standard':
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        elif scaling_method == 'robust':
            numerical_transformer = Pipeline(steps=[('scaler', RobustScaler())])
        elif scaling_method == 'minmax':
            numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    else:
        numerical_transformer = "passthrough"

    # Specify which columns to apply each transformer to
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Include the preprocessor in your main pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline


# In[59]:


categorical_cols = ['SEX','EDUCATION','MARRIAGE','PAY_SEPT','PAY_AUG','PAY_JUL','PAY_JUN','PAY_MAY','PAY_APR']
numerical_cols = [col for col in X_smote.columns if col not in categorical_cols]

# Applying the pipeine to categorical and numerical columns if required
pipeline = create_custom_pipeline(categorical_cols, numerical_cols,scaling_method='robust')


# Fit and transform the data
X_preprocessed_train = pipeline.fit_transform(X_smote)
X_preprocessed_test = pipeline.transform(X_test)


# # 4. Building the Model

# In[69]:


def evaluate_model(model, X_train, X_test, y_train, y_test, hyperparameters={}):
    """
    Evaluate a machine learning model with hyperparameters.

    Parameters:
    - model: The machine learning model to be trained and evaluated.
    - X_train: Features of the training set.
    - X_test: Features of the testing set.
    - y_train: Target labels of the training set.
    - y_test: Target labels of the testing set.
    - hyperparameters: Dictionary of hyperparameters for the model.

    Returns:
    - A dictionary containing evaluation metrics for both training and testing sets, as well as predicted labels.
    """
    # Initialize and configure the model with hyperparameters
    model.set_params(**hyperparameters)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on both training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics for training and testing sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_roc_score = roc_auc_score(y_train, y_train_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_score = roc_auc_score(y_test, y_test_pred)

    # Create a dictionary to store the results
    evaluation_results = {
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1 Score': train_f1,
        'Train ROC AUC Score': train_roc_score,
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1 Score': test_f1,
        'Test ROC AUC Score': test_roc_score,
        'y_pred_train': y_train_pred,
        'y_pred_test': y_test_pred,
        'trained_model':model
    }

    return evaluation_results


# In[70]:


def plot_confusion_matrix(y_true, y_pred, labels = ['Not Defaulter', 'Defaulter']):
    """
    Plots a confusion matrix using seaborn.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of class labels.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Display the plot
    plt.show()


# ### Building a Logistic Regression Model

# In[71]:


logistic_model = LogisticRegression()
results_lr = evaluate_model(logistic_model,X_preprocessed_train, X_preprocessed_test, y_smote, y_test)
results_lr


# In[72]:


plot_confusion_matrix(y_test, results_lr['y_pred_test'])


# ### Building a Random Forest Model

# In[ ]:


Random_forest = RandomForestClassifier()
results_rf = evaluate_model(Random_forest,X_smote, X_test, y_smote, y_test)
results_rf


# In[ ]:


plot_confusion_matrix(y_test, results_rf['y_pred_test'])


# ### Hyper Parameter Tuning for Logistic Regression

# In[ ]:


pipeline = create_custom_pipeline(categorical_cols, numerical_cols,scaling_method='robust')

# Fit and transform the data
X_preprocessed_train = pipeline.fit_transform(X_smote)
X_preprocessed_test = pipeline.transform(X_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[ ]:


#Applying GridSeachCV method

param_grid = {'penalty':['l1','l2','elasticnet', None], 'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver' :['lbfgs', 'liblinear', 'sag', 'saga']}
grid_lr_clf = GridSearchCV(LogisticRegression(), param_grid, scoring = 'accuracy', n_jobs = -1, verbose = 4, cv = 5)
grid_lr_clf.fit(X_preprocessed_train, y_smote)
optimized_clf = grid_lr_clf.best_estimator_
optimized_params = grid_lr_clf.best_params_


# In[ ]:


#Applying RandomSearchCV method


param_grid = {'penalty':['l1','l2','elasticnet', None], 'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver' :['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
random_lr_clf = RandomizedSearchCV(LogisticRegression(), param_distributions=param_grid, scoring='accuracy', n_jobs=-1, verbose=4, cv=5, n_iter=10)
random_lr_clf.fit(X_preprocessed_train, y_smote)
optimized_clf = random_lr_clf.best_estimator_
optimized_params = random_lr_clf.best_params_


# In[ ]:


#Applying Hyperparameter tuning  on Logistic Regression

logistic_model = LogisticRegression()
hp = optimized_params
results_lrh = evaluate_model(model=logistic_model, X_train=X_preprocessed_train, X_test=X_preprocessed_test, y_train=y_smote, y_test=y_test, hyperparameters=hp)
results_lrh


# In[ ]:


plot_confusion_matrix(y_test, results_lrh['y_pred_test'])


# ### Hyper Parameter Tuning for Random Forest

# In[ ]:


# Define the hyperparameter grid
param_dist = {
    'n_estimators': randint(10, 200),  # Number of trees in the forest
    'max_depth': randint(1, 20),      # Maximum depth of each tree
    'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 20),   # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for splitting
    'bootstrap': [True, False],  # Whether or not to use bootstrap samples
    'criterion': ['gini', 'entropy']  # Split criterion
}

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf_classifier, param_distributions=param_dist,
    n_iter=10,  # Number of parameter combinations to try
    scoring='accuracy',  # Choose an appropriate scoring metric
    cv=5,  # Number of cross-validation folds
    verbose=1,  # Set to 1 for progress updates
    n_jobs=-1  # Use all available CPU cores
)

# Fit the RandomizedSearchCV to your training data
random_search.fit(X_smote, y_smote)

# Get the best hyperparameters
RF_best_params = random_search.best_params_


# In[ ]:


RF_best_params = {'min_samples_split':9,'criterion': 'entropy','max_depth': 19,'max_features': 'sqrt','min_samples_leaf': 1,'n_jobs':-1}


# In[ ]:


#Applying Hyperparameter tuning  on Random Forest

random_forest_model = RandomForestClassifier(random_state=42)
hyper_parameter = RF_best_params
results_rfh = evaluate_model(model=random_forest_model, X_train=X_smote, X_test=X_test, y_train=y_smote, y_test=y_test, hyperparameters=hyper_parameter)
results_rfh


# In[ ]:


plot_confusion_matrix(y_test, results_rfh['y_pred_test'])


# In[ ]:


fpr1, tpr1, _ = roc_curve(y_test, results_lr['y_pred_test'])
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_test, results_rfh['y_pred_test'])
roc_auc2 = auc(fpr2, tpr2)

# Repeat the above steps for model3 and its ROC curve and AUC score
plt.figure(figsize=(10, 6))

plt.plot(fpr1, tpr1, color='b', lw=2, label=f'Logistic Regression (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='r', lw=2, label=f'Random Forest (AUC = {roc_auc2:.2f})')

# Plot the ROC curve for model3 with its AUC score

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[73]:


results_lr['trained_model']


# In[74]:


import dill
data = {"model": results_lr['trained_model'], "pipeline":pipeline}   # change the model and pipeline as required
with open ('model_pipeline_object.joblib', 'wb') as file:
    dill.dump(data, file)


# In[ ]:





# ## Conclusion
# 
# In this project, our primary objective was to develop a predictive model for credit card defaults. We aimed to achieve a high recall score, indicating our ability to accurately identify potential default cases. After a rigorous analysis and model tuning process, we have the following key findings and insights:
# 
# - **Model Performance**: We encountered challenges in achieving our desired recall score, which is crucial in credit card default prediction. Recall is important because it measures our model's ability to correctly identify customers at risk of default, minimizing false negatives. 
# 
# - **Model Selection**: Among the algorithms we explored, Random Forest demonstrated superior performance after hyperparameter tuning. This ensemble method effectively captured complex relationships within the data.
# 
# - **Data Limitations**: A critical limitation in our analysis was the absence of essential features, such as customer income, credit scores, and other socio-economic factors. These missing features can significantly impact the predictive power of our model.
# 
# Moving forward, several strategies can be employed to enhance our model's predictive accuracy:
# 
# ### 1. Feature Engineering
# 
# We should explore additional feature engineering techniques to create new variables that could better capture the credit card usage patterns. Some potential features include:
# 
# - **Credit Utilization Ratio**: Calculated as the ratio of credit card balance to the credit limit. High utilization may 
# - **Debt-to-Income Ratio**: A customer's debt relative to their income, reflecting financial stability.
# 
# ### 2. Data Enrichment
# 
# We can enhance the dataset by incorporating external data sources, including credit scores, income data, and employment history. These data points are highly predictive of credit card defaults and can improve our model's performance.
# 
# 
# ### 3. Advanced Algorithms
# 
# While Random Forest has performed well, we should explore advanced algorithms such as gradient boosting (e.g., XGBoost, LightGBM) and deep learning to capture intricate patterns within the data.
# 

# In[ ]:




