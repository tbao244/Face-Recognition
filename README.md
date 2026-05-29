# Ứng dụng Nhận diện khuôn mặt

Hệ thống nhận diện khuôn mặt theo thời gian thực được xây dựng bằng MobileNetV2, OpenCV và Streamlit.

## Cấu trúc thư mục

```
├── app.py               
├── face_train.ipynb   
├── face_model.h5        
├── class_names.pkl      
└── README.md
```

## Danh sách người có thể nhận diện

| STT | Tên |
|-----|------|
| 1 | BUI DANG KHOI |
| 2 | DANG NGUYEN PHUONG NGHI |
| 3 | HA PHUONG THAO |
| 4 | HOANG BAO TRAN |
| 5 | HOANG BUI TRA MY |
| 6 | LE HUYNH DUC HUY |
| 7 | LE MINH TRIET |
| 8 | LE THAI BAO |
| 9 | LE THI NHU QUYNH |
| 10 | LE TRAN QUY ANH |
| 11 | LE TRONG DAI |
| 12 | MAI HO QUOC TUY |
| 13 | NGUYEN BAO HAN |
| 14 | NGUYEN DONG HAI |
| 15 | NGUYEN HOANG BAO |
| 16 | NGUYEN HUU TOAN |
| 17 | NGUYEN KHAC LUU VU |
| 18 | NGUYEN NGOC KHANH UYEN |
| 19 | NGUYEN NGOC KIM TUYET |
| 20 | NGUYEN THI THANH HA |
| 21 | NGUYEN TRONG MINH |
| 22 | NHAN MANH TUAN |
| 23 | PHAM DUC THANH CONG |
| 24 | PHAM LY BAO LAM |
| 25 | PHAM MAI PHUONG |
| 26 | THAI TUAN PHAT |
| 27 | TRAN GIA HAN |
| 28 | TRAN MINH HOANG |
| 29 | TRAN NGOC THAO ANH |
| 30 | TRAN THE DANG KHOA |
| 31 | TRINH THUY HANG |

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Chi tiết mô hình

| Thuộc tính | Giá trị |
|----------|-------|
| Mô hình nền | MobileNetV2 |
| Kích thước đầu vào | 224 × 224 |
| Số lớp | 31 |
| Bộ nhận diện khuôn mặt | Haar Cascade |
| Ngưỡng tin cậy | 0.35 |

Quá trình huấn luyện được chia làm hai giai đoạn:
- **Phase 1** — đóng băng mô hình nền, chỉ huấn luyện các lớp phía trên trong tối đa 30 epoch.
- **Phase 2** — mở khóa 30 lớp cuối của mô hình nền để fine-tune với learning rate thấp hơn trong tối đa 50 .

## Ghi chú

- Số lượng ảnh không nhiều nên mức độ tin cậy không quá cao, dễ dự đoán sai.
