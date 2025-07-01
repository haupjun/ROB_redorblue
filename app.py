import streamlit as st
from PIL import Image, ImageOps
import torch
from utils import predict_image

st.header("당신의 정치 성향은?", divider = 'gray')
st.subheader("**이미지를 업로드하면 :blue[진보]인지 :red[보수]인지 예측해드려요!**")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image) 
    image = ImageOps.fit(image, (500, 500))
    st.image(image, caption='업로드한 이미지', use_container_width=False)

    if st.button("분석하기"):
        result = predict_image(image)
        if result == "진보":
            st.success(f"당신은 :blue[**{result}**] 성향으로 판단됩니다!")
        else: 
            st.success(f"당신은 :red[**{result}**] 성향으로 판단됩니다!")

# streamlit run app.py