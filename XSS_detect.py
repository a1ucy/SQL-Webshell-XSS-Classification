# web攻击分析模型（包括SQL注入、XSS、WebShell、远程命令/代码执行、CSRF等）
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf

# tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], 'GPU')
df = pd.read_csv('xss_dataset.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# balance data
min_len = df['Label'].value_counts().min()
df_class_0 = df[df['Label'] == 0].sample(min_len)
df_class_1 = df[df['Label'] == 1].sample(min_len)
df = pd.concat([df_class_0, df_class_1])

y = df['Label']
X = df.drop('Label', axis=1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

def tf_model(X_train_shape1):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_shape1,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='BinaryCrossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    # model.summary()
    return model


patience = 3
batch_size = 64
epochs = 10
callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

class MetricsF1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        precision = logs['precision']
        recall = logs['recall']
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(" - F1_train:", round(f1_score, 4))

        val_precision = logs['val_precision']
        val_recall = logs['val_recall']
        val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        print(" - F1_val:", round(val_f1_score, 4))

def train_model(X, y, bs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    model = tf_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, verbose=2, validation_split=0.2,
                        callbacks=[MetricsF1(),callback])
    print('\n Evaluating model...')
    model.evaluate(X_test, y_test, verbose=1)

    # model_path = f'model_xss.h5'
    # model.save(model_path)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


train_model(X_scaled, y_scaled, batch_size)
