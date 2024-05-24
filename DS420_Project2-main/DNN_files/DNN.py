import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import time
from creditcard_preparation import prepare_creditcard_data, combine_algo_and_pipeline, load_creditcard_data





def show_model_results(model):
    pred = model.predict(X_test)
    pred = np.where(pred > 0.5, 1, 0)
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def plot_history(history):
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` 
        # is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined 
        # as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further 
        # defined as "for at least 2 epochs"
        patience=200,
        verbose=1,
    )
]



def create_deep_nnmodel(neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model 





print()
print("Preparing DNN model")

df = load_creditcard_data()

random_state = 123
features = [col for col in df.columns if col not in ['id', 'Class']]
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Class'], test_size=0.2, random_state=random_state, stratify=df['Class'], shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





num_neurons = 800000

deep_nn_model8 = create_deep_nnmodel(num_neurons)

# start_time = time.time()

deep_nn_model8.fit(X_train, y_train, epochs=10, batch_size=512, verbose=1, validation_split=0.2, callbacks=callbacks)

# print(f'training time: {round(time.time()-start_time, 2)} seconds')


dnn_predictions = deep_nn_model8.predict(X_test)

dnn_accuracy = {"Accuracy": accuracy_score(y_test, dnn_predictions)}
dnn_accuracy_df = pd.DataFrame(data=dnn_accuracy,
                               index=["DNN"])