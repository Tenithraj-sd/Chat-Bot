import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import random
import streamlit as st

# Load intents from JSON
def load_intents():
    with open('intents.json', 'r') as file:
        return json.load(file)['intents']

intents = load_intents()

# Preprocess data
patterns = []
tags = []
tag_classes = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    if intent['tag'] not in tag_classes:
        tag_classes.append(intent['tag'])

encoder = LabelEncoder()
encoded_tags = encoder.fit_transform(tags)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Convert labels to categorical
y = tf.keras.utils.to_categorical(encoded_tags, len(tag_classes))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(tag_classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save model and data
model.save("chatbot_model.h5")
with open("encoder_classes.json", "w") as f:
    json.dump(encoder.classes_.tolist(), f)
with open("vectorizer.json", "w") as f:
    json.dump(vectorizer.get_feature_names_out().tolist(), f)

print("Model trained and saved successfully.")

# Load model and preprocessing tools
model = tf.keras.models.load_model("chatbot_model.h5")
with open("encoder_classes.json") as f:
    classes = json.load(f)
with open("vectorizer.json") as f:
    vocab = json.load(f)

vectorizer = CountVectorizer(vocabulary=vocab)

# Load intents again for response generation
intents = load_intents()

# Response function
def get_response(user_input):
    input_data = vectorizer.transform([user_input]).toarray()
    prediction = model.predict(input_data)
    tag = classes[np.argmax(prediction)]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Streamlit Interface
st.title("Chatbot")
st.write("Hello! Ask me anything.")

if 'responses' not in st.session_state:
    st.session_state.responses = []

# Handle input and display chat history
def process_input():
    user_input = st.session_state.user_input
    response = get_response(user_input)
    st.session_state.responses.append((user_input, response))
    st.session_state.user_input = ""

st.text_input("You: ", key="user_input", on_change=process_input)

for user_msg, bot_msg in reversed(st.session_state.responses):
    st.write(f"You: {user_msg}")
    st.write(f"Bot: {bot_msg}")

