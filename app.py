import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Setting untuk memilih algoritma
algorithms = ['Pilih Algoritma', 'Support Vector Machine (SVM)', 'Logistic Regression (LR)']
selected_algorithm = st.selectbox('Pilih Algoritma', algorithms)

# Memastikan hanya konten sesuai algoritma yang dipilih yang ditampilkan
if selected_algorithm != 'Pilih Algoritma':
    # Menampilkan algoritma yang dipilih
    st.write(f'Anda memilih algoritma: {selected_algorithm}')

    # Load dataset
    df = pd.read_csv('diabetes.csv')

    # Pisahkan fitur dan target
    X = df.drop(columns='Outcome')
    y = df['Outcome']

    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menampilkan penjelasan atau bagian terkait dengan algoritma yang dipilih
    if selected_algorithm == 'Support Vector Machine (SVM)':
        st.write("Implementasi Support Vector Machine (SVM) digunakan untuk klasifikasi dengan memisahkan data menggunakan hyperplane yang optimal.")
        
        # Menampilkan gambar SVM
        st.image('header_image.png', caption='Support Vector Machine (SVM)', use_container_width=True)
        
        # Membaca model SVM dari pickle
        try:
            svm_model = pickle.load(open('svm_model.sav', 'rb'))
            st.write("Model SVM berhasil dimuat!")
        except FileNotFoundError:
            st.write("Model SVM tidak ditemukan! Latih model terlebih dahulu.")
            # Latih model SVM jika model tidak ditemukan
            svm_model = SVC()
            svm_model.fit(X_train, y_train)
            pickle.dump(svm_model, open('svm_model.sav', 'wb'))
            st.write("Model SVM berhasil dilatih dan disimpan.")
        
        # Prediksi dan evaluasi model SVM
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        st.write(f"Akurasi model SVM: {svm_accuracy:.2f}")
        
        # Form input untuk prediksi baru
        st.subheader('Prediksi Diabetes Baru dengan SVM')
        
        pregnancies = st.number_input('Pregnancies', min_value=0, value=0)
        glucose = st.number_input('Glucose', min_value=0, value=120)
        blood_pressure = st.number_input('BloodPressure', min_value=0, value=70)
        skin_thickness = st.number_input('SkinThickness', min_value=0, value=20)
        insulin = st.number_input('Insulin', min_value=0, value=80)
        bmi = st.number_input('BMI', min_value=0.0, value=25.0)
        diabetes_pedigree = st.number_input('DiabetesPedigreeFunction', min_value=0.0, value=0.5)
        age = st.number_input('Age', min_value=0, value=30)

        # Tombol untuk prediksi
        if st.button('Prediksi dengan SVM'):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], 
                            columns=X.columns)
            prediction = svm_model.predict(input_data)
            
            if prediction[0] == 1:
                st.success("Hasil Prediksi: Diabetes")
            else:
                st.success("Hasil Prediksi: Tidak Diabetes")

    elif selected_algorithm == 'Logistic Regression (LR)':
        st.write("Implementasi Logistic Regression (LR) digunakan untuk klasifikasi biner dengan model berbasis probabilitas.")
        
        # Menampilkan gambar Logistic Regression
        st.image('header_image.png', caption='Logistic Regression (LR)', use_container_width=True)
        
        # Inisialisasi dan latih model Logistic Regression
        lr_model = LogisticRegression(max_iter=200)
        lr_model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi model Logistic Regression
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        st.write(f"Akurasi model Logistic Regression: {lr_accuracy:.2f}")
        
        # Confusion Matrix
        st.subheader('Confusion Matrix')
        conf_matrix = confusion_matrix(y_test, lr_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        
        # Classification Report
        st.subheader('Classification Report')
        st.text(classification_report(y_test, lr_pred))
        
        # Form input untuk prediksi baru
        st.subheader('Prediksi Diabetes Baru dengan Logistic Regression')
        
        pregnancies = st.number_input('Pregnancies', min_value=0, value=0)
        glucose = st.number_input('Glucose', min_value=0, value=120)
        blood_pressure = st.number_input('BloodPressure', min_value=0, value=70)
        skin_thickness = st.number_input('SkinThickness', min_value=0, value=20)
        insulin = st.number_input('Insulin', min_value=0, value=80)
        bmi = st.number_input('BMI', min_value=0.0, value=25.0)
        diabetes_pedigree = st.number_input('DiabetesPedigreeFunction', min_value=0.0, value=0.5)
        age = st.number_input('Age', min_value=0, value=30)

        # Tombol untuk prediksi
        if st.button('Prediksi dengan Logistic Regression'):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], 
                            columns=X.columns)
            prediction = lr_model.predict(input_data)
            
            if prediction[0] == 1:
                st.success("Hasil Prediksi: Diabetes")
            else:
                st.success("Hasil Prediksi: Tidak Diabetes")

