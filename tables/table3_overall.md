| Model | Precision | Recall | F1-score | Accuracy | AP50 (COCO) | FPS (RTX 3080 Ti) |
|---|---|---|---|---|---|---|
| Faster R-CNN | 0.974 | 0.972 | 0.973 | 0.973 | 53.1 | 8.6 |
| Mask R-CNN | 0.933 | 0.927 | 0.930 | 0.939 | 39.4 | 13.2 |
| RetinaNet | 0.968 | 0.963 | 0.965 | 0.966 | 56.9 | 10.8 |
| YOLOv8-seg | 0.947 | 0.942 | 0.944 | 0.952 | 37.7 | 33.9 |
| YOLOv10 | 0.973 | 0.972 | 0.972 | 0.973 | 50.1 | 53.9 |

Precision/Recall/F1 là trung bình không trọng số của hai lớp; Accuracy là tỷ lệ đường chéo của ma trận 2×2 (công thức 4). AP50 tính theo chuẩn COCO trên cùng tập test, không dùng ngưỡng conf 0,30.
