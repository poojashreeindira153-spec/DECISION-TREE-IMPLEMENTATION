#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[12]:


data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Shape of dataset:", X.shape)
X.head()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[8]:


model=DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.show()


# In[10]:


plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=data.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()


# In[11]:


print("""Observations:
1. The tree splits mainly using petal length and width.
2. Accuracy is good on the test dataset.
3. Limiting max_depth avoids overfitting.
4. Model can be tuned with different criteria (gini/entropy).
""")

