# Epochs Sayısını Belirlemek

## Epochs ve Batch Size

- **Epochs**: Modelin tüm eğitim verilerini kaç kez işleyeceğini belirler.
- **Batch Size**: Modelin parametrelerini güncellemeden önce kaç veri örneği üzerinde hesaplama yapacağını belirler.

### Epochs Sayısını Belirlemek

Epochs sayısını belirlerken dikkat edilmesi gerekenler:

- **Az epoch sayısı**: Modelin yeterince öğrenememesine (underfitting) neden olabilir.
- **Çok epoch sayısı**: Modelin eğitim verilerine aşırı uyum göstermesine (overfitting) yol açabilir.

### Batch Size'ı Belirlemek

Batch size'ın etkileri:

- **Küçük batch size**: Daha fazla güncelleme ve daha yüksek hesaplama maliyeti. Ancak, model daha genel bir çözüm bulabilir.
- **Büyük batch size**: Daha az güncelleme ve daha düşük hesaplama maliyeti. Ancak, model daha spesifik bir çözüm bulabilir.

## Model Eğitimi İzleme

Model eğitiminizi izlemek için eğitim ve doğrulama kayıplarını ve doğruluklarını grafikleştirebilirsiniz.

### Örnek Kod

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
```

### Model Eğitim ve İzleme

```python
import tensorflow as tf

# Örnek bir model tanımla
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit ve history nesnesini al
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Eğitim sonuçlarını görselleştir
plot_training_history(history)
```

## Early Stopping Kullanımı

`EarlyStopping` callback'i kullanarak modelinizin doğrulama kaybı iyileşmediğinde eğitimi durdurabilirsiniz.

### EarlyStopping Callback'ini Tanımlama

```python
from tensorflow.keras.callbacks import EarlyStopping

# EarlyStopping callback'ini tanımla
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

### Modeli Eğitme ve EarlyStopping Kullanma

```python
# Modeli eğit ve history nesnesini al
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)
```

### Eğitim Sonuçlarını Görselleştirme

```python
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_training_history(history)
```

### Açıklamalar

- **EarlyStopping callback'i**: `monitor` parametresi, hangi metriği izleyeceğinizi belirtir (örneğin, `val_loss`). `patience` parametresi, doğrulama kaybı iyileşmediğinde kaç epoch bekleyeceğinizi belirtir. `restore_best_weights=True`, eğitimin sonunda en iyi model ağırlıklarını geri yükler.
- **callbacks parametresi**: `model.fit` fonksiyonunda, `callbacks` parametresi ile early stopping callback'ini ekleyin.

### Kodun Tamamı

```python
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# EarlyStopping callback'ini tanımla
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Örnek bir model tanımla
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit ve history nesnesini al
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Eğitim sonuçlarını görselleştir
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_training_history(history)
```