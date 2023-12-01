#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd



# In[4]:


data=pd.read_csv("loan-test.csv")
data


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


# check the presence of missing value
data.isna().sum()


# In[10]:


data['Gender'].fillna(
    value=data['Gender'].mode()[0]
)


# In[11]:


data['LoanAmount'].fillna(
    value=data['LoanAmount'].median()
)


# In[12]:


df = data.drop_duplicates()
df


# In[13]:


if df.isnull().any().any():
    df1 = data.dropna()


# In[14]:


df1


# In[15]:


# Extraction of IV and DV
X=data.iloc[:, 0:2]
Y=data.iloc[:, 2:3]
print(X)
print(Y)


# In[16]:


# Before splitting
print("Shape of X-->",X.shape )
print("Shape of Y-->",Y.shape )


# In[17]:


# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)


# In[18]:


X_train


# In[19]:


X_test


# In[20]:


Y_train


# In[21]:


Y_test


# In[22]:


# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Print or use the list of categorical columns
print("Categorical columns:", categorical_columns)


# In[23]:


X.columns


# In[24]:


# Assuming df is your DataFrame and 'column_name' is the column with strings
df['Loan_ID'] = df['Loan_ID'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
df['Loan_ID']


# In[25]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Assuming X is your DataFrame
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numeric_cols = list(X.columns.difference(categorical_cols))

# Create transformers for scaling and one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Assuming X is your DataFrame
X_transformed = preprocessor.fit_transform(X)


# In[26]:


X_transformed


# In[27]:


# Create the pipeline with the preprocessor and linear regression model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# In[28]:


pipeline


# In[29]:


from sklearn.linear_model import LinearRegression
# Assuming 'binary_cols' is a list of column names that contain 'Yes' and 'No'
binary_cols = 'Married'

# Create a mapping for 'Yes' to 1 and 'No' to 0
mapping = {'Yes': 1, 'No': 0}


# Apply the mapping to the specified column in Y_train
Y_train = Y_train.replace(mapping)


# In[30]:


Y_train


# In[31]:


from sklearn.linear_model import LinearRegression
# Assuming 'binary_cols' is a list of column names that contain 'Yes' and 'No'
binary_cols = 'Married'

# Create a mapping for 'Yes' to 1 and 'No' to 0
mapping = {'Yes': 1, 'No': 0}


# Apply the mapping to the specified column in Y_train
Y_test = Y_test.replace(mapping)


# In[32]:


Y_test


# In[33]:


binary_cols = 'Gender'
mapping = {'Male':1, 'Female':0}

X_train = X_train.replace(mapping)


# In[34]:


X_train


# In[35]:


pipeline.fit(X_train, Y_train)


# In[36]:


coefficients = pipeline.named_steps['regressor'].coef_
coefficients


# In[37]:


from sklearn.preprocessing import LabelEncoder


# In[38]:


binary_cols = [ 'Gender']


# In[39]:


label_encoder = LabelEncoder()


# In[40]:


for col in binary_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])


# In[41]:


X_train[col]


# In[42]:


pipeline.fit(X_train, Y_train)


# In[43]:


Y_pred = pipeline.predict(X_train)


# In[44]:


# Assuming you have trained your model and obtained predictions Y_pred

# Print shapes to debug the issue
print("Shape of Y_test:", Y_test.shape)
print("Shape of Y_pred:", Y_pred.shape)


# In[45]:


print("Y_test values:", Y_test)
print("Y_pred values:", Y_pred)


# In[46]:


# Evaluation metrics on the training set


# In[47]:


from sklearn.metrics import mean_absolute_error

# Assuming Y_train and Y_pred are your actual and predicted values on the training set
mae_train = mean_absolute_error(Y_train, Y_pred)

print("Mean Absolute Error on the training set:", mae_train)


# In[48]:


from sklearn.metrics import mean_squared_error

# Assuming Y_train and Y_pred are your actual and predicted values on the training set
mse_train = mean_squared_error(Y_train, Y_pred)

print("Mean Squared Error on the training set:", mse_train)


# In[49]:


from sklearn.metrics import r2_score

# Assuming Y_train and Y_pred are your actual and predicted values on the training set
r2_train = r2_score(Y_train, Y_pred)

print("R-squared on the training set:", r2_train)


# In[ ]:





# In[50]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming Y_test and Y_pred are your actual and predicted values on the test set
print("Length of Y_test:", len(Y_test))
print("Length of Y_pred:", len(Y_pred))

# Check if the lengths are the same
if len(Y_test) != len(Y_pred):
    print("Error: Lengths of Y_test and Y_pred are not the same.")
else:
    mae_test = mean_absolute_error(Y_test, Y_pred)
    mse_test = mean_squared_error(Y_test, Y_pred)
    r2_test = r2_score(Y_test, Y_pred)

    print("Mean Absolute Error on the test set:", mae_test)
    print("Mean Squared Error on the test set:", mse_test)
    print("R-squared on the test set:", r2_test)


# In[ ]:




