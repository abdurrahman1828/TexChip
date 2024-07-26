import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Train a Random Forest model
# Load your dataset
data = pd.read_csv('data/EnvivaWhole_1_train.csv')
test = pd.read_csv('data/EnvivaWhole_1_test.csv')

# Split into features and target
X = data.drop(columns=['label'])
y = data['label']

# Split into features and target
X_t = test.drop(columns=['label'])
y_t = test['label']



# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Step 2: Instantiate LIME explainer
explainer = LimeTabularExplainer(X_t.values,
                                 feature_names=X_t.columns.tolist(),
                                 class_names=np.unique(y_t),
                                 mode='classification')

# Step 3: Explain predictions
def map_moisture_level(array):
    categories = ['dry', 'medium', 'wet']
    max_index = np.argmax(array)
    return categories[max_index]


for i in X_t.index:
    instance_idx = i # Example: explaining the first instance in the dataset

    # Get prediction for the chosen instance
    prediction = rf_model.predict_proba(X_t.iloc[[instance_idx]])[0]
    out = map_moisture_level(prediction)
    if out!=y_t[instance_idx]:
        print(i)
        print(prediction, y_t[instance_idx])


    # Explain the prediction
    explanation = explainer.explain_instance(X_t.iloc[instance_idx].values,
                                             rf_model.predict_proba,
                                             num_features=len(X.columns))

    ###print(explanation.as_list())


    ### Get feature names and their weights from the explanation
    features = explanation.as_list()[::-1]

    feature_conditions = [f[0] for f in features]
    weights = [f[1] for f in features]

    ###print(feature_names_sorted)

    feature_names = [f[0] for f in features][1:]
    weights = [f[1] for f in features][1:]

    ##### Set the number of xticks
    plt.figure(figsize=(3,6))
    num_xticks = 3  # Set the desired number of ticks
    plt.gca().locator_params(axis='x', nbins=num_xticks + 1)
    plt.barh(feature_names, weights, color='maroon', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Feature')
    #plt.tick_params(axis='y', labelsize=10, pad=10)
    plt.savefig(f'plots/lime_plots/enviva/{str(y_t[instance_idx])}_test_ind_{instance_idx}.jpg', dpi = 600, bbox_inches= 'tight')
    plt.close()

    # Labels for the categories
    categories = ['dry', 'medium', 'wet']

    # Creating a small bar plot
    plt.figure(figsize=(2, 1.5))  # Set the figure size to be small
    plt.bar(categories, prediction, color='maroon',width=0.5, edgecolor='black')

    # Set the labels and ticks
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.savefig(f'plots/lime_plots/enviva/{str(y_t[instance_idx])}_test_ind_{instance_idx}_hist.jpg', dpi=300,
                bbox_inches='tight')
    plt.close()





