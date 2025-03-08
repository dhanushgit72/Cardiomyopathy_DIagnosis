from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
MODEL_PATH = r"C:\Major Project\model\heart_model.h5"
TEST_DATA_DIR = r"C:\Major Project\train_test_split/test_data"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

# Function to find most similar images
def find_most_similar(test_image_path, compare_dir, category):
    test_img = cv2.imread(test_image_path)
    test_img = cv2.resize(test_img, (150, 150))
    test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_similarity = -1
    best_image_path = ""

    for img_name in os.listdir(compare_dir):
        img_path = os.path.join(compare_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        similarity = ssim(test_img_gray, img_gray)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = img
            best_image_path = img_path

    # Save best match to static/results folder
    if best_match is not None:
        save_path = os.path.join(RESULTS_FOLDER, f"best_{category}.jpg")
        cv2.imwrite(save_path, best_match)
        return f"/static/results/best_{category}.jpg", best_similarity * 100  # Return web-accessible path
    return None, 0

@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) / 255.0

            # Predict
            prediction = model.predict(img_array)[0][0]
            result = "Sick" if prediction > 0.5 else "Healthy"

            # Find most similar images
            healthy_img, healthy_sim = find_most_similar(filepath, os.path.join(TEST_DATA_DIR, "Normal"), "healthy")
            sick_img, sick_sim = find_most_similar(filepath, os.path.join(TEST_DATA_DIR, "Sick"), "sick")

            # Compute difference heatmap
            test_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            sick_img_data = cv2.imread(sick_img.replace("/static", "static"), cv2.IMREAD_GRAYSCALE)

            if test_img is not None and sick_img_data is not None:
                test_img = cv2.resize(test_img, (150, 150))
                sick_img_data = cv2.resize(sick_img_data, (150, 150))

                diff = cv2.absdiff(test_img, sick_img_data)
                diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                diff_colored = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)

                # Save heatmap in /static/results
                diff_path = os.path.join(RESULTS_FOLDER, "diff_heatmap.jpg")
                cv2.imwrite(diff_path, diff_colored)
                diff_path = "/static/results/diff_heatmap.jpg"  # Web-accessible path
            else:
                diff_path = None

            return render_template("result.html", 
                                   result=result, 
                                   test_img=f"/static/uploads/{filename}",
                                   healthy_img=healthy_img,
                                   healthy_sim=healthy_sim,
                                   sick_img=sick_img,
                                   sick_sim=sick_sim,
                                   diff_img=diff_path)
    return render_template("index.html")

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/results/<filename>")
def results_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
