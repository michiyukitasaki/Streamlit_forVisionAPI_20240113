import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import base64
import math
import shutil
import tempfile
import os
import zipfile

api_key = "AIzaSyAwmqgp-TzVphXOI6bFpwRTxE8BDuRk12E"

def calculate_dimensions(vertices, calibration_factor):
    width = math.sqrt((vertices[1]['x'] - vertices[0]['x'])**2 + (vertices[1]['y'] - vertices[0]['y'])**2) * calibration_factor
    height = math.sqrt((vertices[2]['x'] - vertices[1]['x'])**2 + (vertices[2]['y'] - vertices[1]['y'])**2) * calibration_factor
    return round(max(width, height), 2), round(min(width, height), 2)

def draw_bounding_box(image, vertices, text=None):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    points = [(vertex['x'] * width, vertex['y'] * height) for vertex in vertices]
    draw.polygon(points, outline="red")
    if text:
        font = ImageFont.load_default()
        draw.text((points[0][0], points[0][1]), text, fill="red", font=font)
    return image

def process_image(image):
    api_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    headers = {"Content-Type": "application/json"}
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode()
    payload = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": "OBJECT_LOCALIZATION"}]
            }
        ]
    }
    response = requests.post(api_url, json=payload, headers=headers)
    objects = response.json()['responses'][0]['localizedObjectAnnotations']
    return objects

st.title('画像解析アプリ')
calibration_value = st.number_input("キャリブレーション用の長さ(cm)", min_value=0.0, value=10.0, step=0.1)
uploaded_files = st.file_uploader("画像を複数アップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
analyze_button = st.button('解析開始')

if analyze_button and uploaded_files:
    data = []
    images_to_show = []
    calibration_factor = None
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        objects = process_image(image)
        if objects:
            vertices = objects[0]['boundingPoly']['normalizedVertices']
            if idx == 0:
                real_length = calibration_value
                pixel_length = max(math.sqrt((vertices[1]['x'] - vertices[0]['x'])**2 + (vertices[1]['y'] - vertices[0]['y'])**2),
                                   math.sqrt((vertices[2]['x'] - vertices[1]['x'])**2 + (vertices[2]['y'] - vertices[1]['y'])**2))
                calibration_factor = real_length / pixel_length
                image_with_box = draw_bounding_box(image.copy(), vertices, text="Calibration Image")
            else:
                image_with_box = draw_bounding_box(image.copy(), vertices)
            long_side, short_side = calculate_dimensions(vertices, calibration_factor)
            data.append({"filename": uploaded_file.name, "長辺(cm)": long_side, "短辺(cm)": short_side})
            images_to_show.append(image_with_box)

    for i in range(0, len(images_to_show), 4):
        cols = st.columns(4)
        for col, image in zip(cols, images_to_show[i:i+4]):
            col.image(image, width=150)

    df = pd.DataFrame(data)
    stats = df.describe().round(2) if not df.empty else pd.DataFrame()

    # 一時ファイルの作成
    with tempfile.TemporaryDirectory() as tmpdir:
        df_path = os.path.join(tmpdir, 'result.csv')
        stats_path = os.path.join(tmpdir, 'statistics.csv')
        zip_path = os.path.join(tmpdir, 'data.zip')

        # CSVファイルの保存
        df.to_csv(df_path, index=False)
        stats.to_csv(stats_path)

        # ZIPファイルの作成
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(df_path, 'result.csv')
            zipf.write(stats_path, 'statistics.csv')

        # ZIPファイルのダウンロード
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="Download CSV files as ZIP",
                data=f,
                file_name='data.zip',
                mime='application/zip'
            )
