import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import xgboost as xgb

from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


@st.cache_resource
def load_dl_model(path):
    return tf.keras.models.load_model(path, custom_objects={"mse": custom_mse})

@st.cache_resource
def load_xgboost_model(path):
    model = xgb.Booster()
    model.load_model(path)
    
    return model

dl_price_model = load_dl_model("model/dl_model_price_prediction.h5")
dl_delivery_model = load_dl_model("model/dl_model_delivery_prediction.h5")

xgb_price_model = load_xgboost_model("model/xgboost_model_price_prediction.json")
xgb_delivery_model = load_xgboost_model("model/xgboost_model_delivery_prediction.json")

st.title("Prediksi E-Commerce")

prediction_type = st.sidebar.selectbox(
    "Pilih Jenis Prediksi", 
    ["Prediksi Harga Barang", "Prediksi Lama Pengiriman"]
)

if prediction_type == "Prediksi Harga Barang":
    st.header("Prediksi Harga Barang")
    
    product_category_encoder = np.load("data/fix/product_category_name_encoder.npy",allow_pickle=True).tolist()
    product_category_decoder = {v:k for k, v in product_category_encoder.items()}

    
    product_category_name = st.selectbox("Kategori Produk", product_category_decoder.keys())
    product_weight_g = st.number_input("Berat Produk (gram)", min_value=0.0)
    product_length_cm = st.number_input("Panjang Produk (cm)", min_value=0.0)
    product_height_cm = st.number_input("Tinggi Produk (cm)", min_value=0.0)
    product_width_cm = st.number_input("Lebar Produk (cm)", min_value=0.0)
    
    if st.button("Prediksi Harga"):
        input_data = np.array([[
            product_category_decoder[product_category_name],
            product_weight_g,
            product_length_cm,
            product_height_cm,
            product_width_cm
        ]])
        
        print(input_data)
        print("masukkkkk")

        dl_price_pred = dl_price_model.predict(input_data)[0][0]
        dmatrix_input = xgb.DMatrix(input_data)
        xgb_price_pred = xgb_price_model.predict(dmatrix_input)[0]

        st.write("### Prediksi Harga Barang")
        st.write(f"- **Deep Learning Model**: USD {dl_price_pred:,.2f}")
        st.write(f"- **XGBoost Model**: USD {xgb_price_pred:,.2f}")

elif prediction_type == "Prediksi Lama Pengiriman":
    st.header("Prediksi Lama Pengiriman")
    
    seller_customer_distance_km = st.number_input("Jarak Penjual-Pembeli (km)", min_value=0.0)
    price = st.number_input("Harga Barang (USD)", min_value=0.0)
    freight_value = st.number_input("Nilai Ongkir (USD)", min_value=0.0)
    product_weight_g = st.number_input("Berat Produk (gram)", min_value=0.0)
    product_length_cm = st.number_input("Panjang Produk (cm)", min_value=0.0)
    product_height_cm = st.number_input("Tinggi Produk (cm)", min_value=0.0)
    product_width_cm = st.number_input("Lebar Produk (cm)", min_value=0.0)

    if st.button("Prediksi Lama Pengiriman"):
        input_data = np.array([[
            seller_customer_distance_km,
            price,
            freight_value,
            product_weight_g,
            product_length_cm,
            product_height_cm,
            product_width_cm
        ]])

        dl_delivery_pred = dl_delivery_model.predict(input_data)[0][0]
        dmatrix_input = xgb.DMatrix(input_data)
        xgb_delivery_pred = xgb_delivery_model.predict(dmatrix_input)[0]

        st.write("### Prediksi Lama Pengiriman")
        st.write(f"- **Deep Learning Model**: {int(dl_delivery_pred)} hari")
        st.write(f"- **XGBoost Model**: {int(xgb_delivery_pred)} hari")
