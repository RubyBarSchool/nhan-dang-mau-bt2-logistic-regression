# Bài tập 2 — Cài đặt và đánh giá Logistic Regression

Môn Nhận dạng mẫu (TS. Nguyễn Xuân Đức), Học viện Công nghệ Bưu chính Viễn thông.
Học viên: Trần Vũ Trường — B26CHKH096 — lớp M26CQKH02-B.

Logistic Regression nhị phân cài bằng NumPy (score, sigmoid, log loss, gradient descent),
so sánh với `sklearn.linear_model.LogisticRegression` trên bộ dữ liệu
[Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545) của UCI:
3.810 hạt gạo, 7 đặc trưng hình thái, 2 giống.

## Kết quả chính (tập Test, 762 mẫu)

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss |
|---|---|---|---|---|---|---|
| Baseline (theo tỷ lệ lớp) | 0,5722 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,6827 |
| NumPy, gradient descent | 0,9239 | 0,8964 | 0,9294 | 0,9127 | 0,9814 | 0,1767 |
| sklearn LogisticRegression | 0,9239 | 0,8964 | 0,9294 | 0,9127 | 0,9814 | 0,1767 |

Hai bản tối ưu cùng một hàm mục tiêu (L2 với λ = 1/(2Cn), C = 1) nên cho 762/762 dự đoán
giống nhau; xác suất lệch tối đa 3 × 10⁻⁵. Gradient descent cần 15.476 vòng lặp, lbfgs của
sklearn cần 43 vòng, Newton tự cài cần 6 vòng.

## Cấu trúc

```
B26CHKH096_TranVuTruong_BaiTap2.ipynb   notebook đã chạy, có đầy đủ output
B26CHKH096_TranVuTruong_BaiTap2.pdf     báo cáo
src/logreg_numpy.py                     cài đặt NumPy (gradient descent và Newton)
data/Rice_Cammeo_Osmancik.arff          dữ liệu gốc từ UCI
outputs/results.json                    mọi con số trong báo cáo
outputs/figures/                        hình vẽ
report/                                 mã sinh báo cáo từ results.json
```

## Cách chạy

Cần Python 3.12.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace B26CHKH096_TranVuTruong_BaiTap2.ipynb
```

Hoặc mở notebook và chọn *Restart & Run All*. Notebook chạy trong khoảng một phút, không cần
tải thêm dữ liệu.

Dựng lại báo cáo (Windows, dùng Microsoft Edge để in PDF):

```bash
python report/build_report.py
msedge --headless=new --no-pdf-header-footer --print-to-pdf=B26CHKH096_TranVuTruong_BaiTap2.pdf report/report.html
```

## Nguồn dữ liệu

Cinar, I., Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence
Methods. *International Journal of Intelligent Systems and Applications in Engineering*, 7(3), 188–194.

Mã nguồn: https://github.com/RubyBarSchool/nhan-dang-mau-bt2-logistic-regression
