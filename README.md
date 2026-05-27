# Nhận diện khuôn mặt

Ứng dụng nhận diện khuôn mặt theo thời gian thực được xây dựng bằng MobileNetV2 và Streamlit. Mô hình được huấn luyện bằng phương pháp transfer learning trên tập dữ liệu tùy chỉnh và thực hiện nhận diện trực tiếp qua webcam.

## Cấu trúc thư mục

```
├── app.py              
├── face_train.ipynb     
├── face_model.h5        
└── class_names.pkl      
```

## Chi tiết mô hình

**Huấn luyện mô hình** 
- Sử dụng **MobileNetV2** pretrained trên ImageNet làm model gốc 
- Thêm custom head: `GlobalAveragePooling2D → BatchNorm → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax`
- Kích thước đầu vào: `224×224` pixels
- Data augmentation: rotation, brightness, zoom, horizontal flip
- Chia dữ liệu train/validation theo tỉ lệ 80/20  
- Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`
- Tên lớp được lưu vào `class_names.pkl`

**Nhận diện** 
- Haar Cascade dùng để phát hiện khuôn mặt trong từng frame webcam
- Khuôn mặt đã nhận diện sẽ được resize về 224×224 trước khi đưa vào model
- Nếu confidence ≥ 65%, hệ thống sẽ hiển thị tên người; thấp hơn ngưỡng này sẽ được gán là Unknown
- FPS, số lượng khuôn mặt và kết quả dự đoán được hiển thị trên giao diện

