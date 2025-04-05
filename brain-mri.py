# %%
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19, ResNet50V2, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
warnings.filterwarnings("ignore")

# %%
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# %%
def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# %%
def FFM(low_level_features, high_level_features, num_filters):
    # Upsampling
    high_level_upsampled = UpSampling2D(size=(2, 2))(high_level_features)
    high_level_upsampled = Conv2D(num_filters, (3, 3), padding="same")(high_level_upsampled)
    high_level_upsampled = BatchNormalization()(high_level_upsampled)
    high_level_upsampled = Activation("relu")(high_level_upsampled)

    # Perform 1x1 convolution on low-level features
    low_level_processed = Conv2D(num_filters, (1, 1), padding="same")(low_level_features)
    low_level_processed = BatchNormalization()(low_level_processed)
    low_level_processed = Activation("relu")(low_level_processed)

    # Combine low-level and high-level features
    #combined = Concatenate()([low_level_processed, high_level_upsampled])
    low_level_resized = UpSampling2D(size=(high_level_upsampled.shape[1] // low_level_processed.shape[1], 
                                       high_level_upsampled.shape[2] // low_level_processed.shape[2]))(low_level_processed)

    # Combine features after ensuring matching shapes
    combined = Concatenate()([low_level_resized, high_level_upsampled])
    
    # Fuse features
    fused = Conv2D(num_filters, (1, 1), padding="same")(combined)
    fused = BatchNormalization()(fused)
    fused = Activation("relu")(fused)

    return fused
# %%
# Load training data
def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])
    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

# Load testing data
def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])
    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

# Loading training and testing data
tr_df = train_df('Training')
ts_df = test_df('Testing')


# %%
# Visualize training data class distribution
plt.figure(figsize=(15,7))
ax = sns.countplot(data=tr_df, y=tr_df['Class'])
plt.title('Count of images in each class', fontsize=20)
ax.bar_label(ax.containers[0])
plt.show()

# Visualize testing data class distribution
plt.figure(figsize=(15, 7))
ax = sns.countplot(y=ts_df['Class'], palette='viridis')
ax.set(title='Count of images in each class')
ax.bar_label(ax.containers[0])
plt.show()


# %%
tst_df, val_df = train_test_split(ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'], shuffle=True)


# %%
batch_size = 16
img_size = (256, 256)

_gen = ImageDataGenerator(
    rescale= 1/255, brightness_range= (0.8, 1.2)
)

test_gen = ImageDataGenerator(rescale=1/255)

tr_gen_data = _gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                       y_col='Class', batch_size=batch_size,
                                       target_size=img_size)
val_gen_data = _gen.flow_from_dataframe(val_df, x_col='Class Path',
                                        y_col='Class', batch_size=batch_size,
                                        target_size=img_size)
test_gen_data= test_gen.flow_from_dataframe(tst_df, x_col='Class Path',
                                            y_col='Class', batch_size=16,
                                            target_size=img_size, shuffle = False)


# %%
class_dict = tr_gen_data.class_indices
classes = list(class_dict.keys())
images, labels = next(tr_gen_data)

plt.figure(figsize=(20, 20))

for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(8,8, i + 1)
    plt.imshow(image)
    class_name = classes[np.argmax(label)]
    plt.title(class_name, color='k', fontsize=15)

plt.show()
# %%
# def build_vgg19_unet(input_shape):
#     inputs = Input(input_shape)
#     vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

#     s1 = vgg19.get_layer("block1_conv2").output
#     s2 = vgg19.get_layer("block2_conv2").output
#     s3 = vgg19.get_layer("block3_conv4").output
#     s4 = vgg19.get_layer("block4_conv4").output

#     f1 = FFM(s1, s2, 64)
#     f2 = FFM(s2, s3, 128)
#     f3 = FFM(s3, s4, 256)

#     # d1 = decoder_block(s5, f4, 512)
#     d2 = decoder_block(s4, f3, 256)
#     d3 = decoder_block(d2, f2, 128)
    
#     x1 = UpSampling2D(size=(2, 2))(d3)
#     # siem = SIEM(x1, f1, 64, 2)
#     combined = Concatenate()([x1, f1])
#     print(combined)
#     outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(combined)

#     model = Model(inputs, outputs, name="VGG19_U-Net")
#     return model4
def build_inception_unet(input_shape):
    inputs = Input(input_shape)
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    # vgg19.load_weights('/kaggle/input/vgg19/tensorflow2/default/1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    s1 = vgg19.get_layer("block1_conv2").output
    s2 = vgg19.get_layer("block2_conv2").output
    s3 = vgg19.get_layer("block3_conv4").output
    s4 = vgg19.get_layer("block4_conv4").output
    
    f1 = FFM(s1, s2, 64)
    f2 = FFM(s2, s3, 128)
    f3 = FFM(s3, s4, 256)
    
    d2 = decoder_block(s4, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    
    x1 = UpSampling2D(size=(2, 2))(d3)
    combined = Concatenate()([x1, f1])
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(combined)
    
    model = Model(inputs, outputs, name="Inception_U-Net")
    return model


# %%

# ResNet Model
input_shape = (256, 256, 3)
base_model = build_inception_unet(input_shape)

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

model.summary()

# %%
history = model.fit(tr_gen_data, epochs=20, validation_data=val_gen_data, shuffle=False)
history.history.keys()

# %%
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
tr_per = history.history['precision']
tr_recall = history.history['recall']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
val_per = history.history['val_precision']
val_recall = history.history['val_recall']


index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
index_precision = np.argmax(val_per)
per_highest = val_per[index_precision]
index_recall = np.argmax(val_recall)
recall_highest = val_recall[index_recall]

Epochs = [i + 1 for i in range(len(tr_acc))]
loss_label = f'Best epoch = {str(index_loss + 1)}'
acc_label = f'Best epoch = {str(index_acc + 1)}'
per_label = f'Best epoch = {str(index_precision + 1)}'
recall_label = f'Best epoch = {str(index_recall + 1)}'


plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')


plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)


plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)


plt.subplot(2, 2, 3)
plt.plot(Epochs, tr_per, 'r', label='Precision')
plt.plot(Epochs, val_per, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
plt.title('Precision and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)


plt.subplot(2, 2, 4)
plt.plot(Epochs, tr_recall, 'r', label='Recall')
plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)


plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()

# %%
train_score = model.evaluate(tr_gen_data, verbose=1)
valid_score = model.evaluate(val_gen_data, verbose=1)
test_score = model.evaluate(test_gen_data, verbose=1)

print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print('-' * 20)
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print('-' * 20)
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")

preds = model.predict(test_gen_data)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(test_gen_data.classes, y_pred)
labels = list(class_dict.keys())
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')
plt.show()


# %%
def predict(img_path):
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((256, 256))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt = '%.2f')
    plt.show()

# Predict for an image
predict('Testing/glioma/Te-glTr_0000.jpg')
predict('Testing/meningioma/Te-meTr_0003.jpg')
predict('Testing/pituitary/Te-piTr_0003.jpg')



