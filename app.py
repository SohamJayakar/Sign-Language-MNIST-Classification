import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image

# Load your trained model
model_path = 'best_model.h5'
weights_path = 'best_model_weights.h5'

model = load_model(model_path)
model.load_weights(weights_path)

# Example labels mapping
labels_mapping = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e',
    5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
    15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
    20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'
}

def preprocess_frame(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (28, 28))
    img_array = np.expand_dims(frame_resized, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def main():
    st.title("Real-Time Sign Language Detection")
    st.write("Using your mobile camera as the webcam")

    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            break

        img_array = preprocess_frame(frame)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = labels_mapping.get(predicted_class[0], 'Unknown')

        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(frame, channels='BGR')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
