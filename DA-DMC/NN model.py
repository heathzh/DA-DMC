from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# upload data
data_path = 'C:/NN model/test/data.csv'  
data = pd.read_csv(data_path)

# data preprocessing
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# split the dataset
X = data.drop('label', axis=1)  
y = data['label']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "C:/NN model/test/scaler.pkl")

y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# build NN model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(y_train_categorical.shape[1], activation='softmax')
])

# Set up early stopping
early_stopping=EarlyStopping(monitor='val_loss',patience=10)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_categorical, epochs=400, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical)
print(f'Accuracy on test set: {accuracy:.2f}')


#Confusion matrix plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Prediction
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.argmax(y_test_categorical, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix heatmap with percentages
print("Starting to plot confusion matrix...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues')
plt.title('Confusion Matrix with Percentages')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

model.save('C:/NN model/test/my_model_saved_model.keras')

