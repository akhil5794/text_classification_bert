#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)

df['reviewText'] = df['reviewText'] + df['summary']
df = df[['reviewText', 'overall']]

df1 = df[df.overall == 5]
df1 = df1.sample(frac=0.38, replace=True, random_state=1)

df = df[df.overall != 5]
df = pd.concat([df, df1], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(df['reviewText'],df['overall'],test_size=0.4)

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='softmax', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])
#model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=METRICS)


model.fit(X_train, y_train, epochs=10)

# **Evaluate model**
#y_predicted = model.predict(X_test)
#y_predicted = y_predicted.flatten()


#import numpy as np
#y_predicted = np.where(y_predicted > 0.5, 1, 0)
#y_predicted