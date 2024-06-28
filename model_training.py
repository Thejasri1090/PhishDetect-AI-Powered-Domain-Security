from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Train Decision Tree classifier
dt_classifier.fit(X_train, y_train)
