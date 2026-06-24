import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# vocabulary size
vocab_size = 5000

# load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# padding
maxlen = 200

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test  = pad_sequences(X_test,  maxlen=maxlen)

# build model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=maxlen))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

# train model
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# save model
model.save("sentiment_model.h5")
print("Model saved")

# ============================================================
#  PRINT FINAL ACCURACY  — copy these numbers into your PPT
# ============================================================

# Evaluate on test data (official score)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n")
print("=" * 50)
print("        FINAL MODEL ACCURACY REPORT")
print("=" * 50)

# Per-epoch results from training history
print("\nEpoch-wise Training Results:")
print("-" * 50)
for i in range(len(history.history['accuracy'])):
    tr_acc  = history.history['accuracy'][i]   * 100
    val_acc = history.history['val_accuracy'][i] * 100
    tr_loss  = history.history['loss'][i]
    val_loss = history.history['val_loss'][i]
    print(f"  Epoch {i+1}:  "
          f"Train Acc = {tr_acc:.2f}%  |  "
          f"Val Acc = {val_acc:.2f}%  |  "
          f"Train Loss = {tr_loss:.4f}  |  "
          f"Val Loss = {val_loss:.4f}")

# Final scores
best_val = max(history.history['val_accuracy']) * 100
final_tr  = history.history['accuracy'][-1]    * 100
final_val = history.history['val_accuracy'][-1] * 100

print("\n" + "-" * 50)
print(f"  Training  Accuracy  (Epoch 3) : {final_tr:.2f}%")
print(f"  Validation Accuracy (Epoch 3) : {final_val:.2f}%")
print(f"  Best Validation Accuracy      : {best_val:.2f}%")
print(f"  Test Loss                     : {test_loss:.4f}")
print(f"  Test Accuracy (model.evaluate): {test_acc * 100:.2f}%")
print("=" * 50)
print("\n  NOTE these numbers for your PPT slide!")
print("=" * 50)