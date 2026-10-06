"""Sinh báo cáo HTML từ outputs/results.json (mọi con số lấy từ kết quả notebook).

    python report/build_report.py            -> report/report.html
Sau đó in ra PDF bằng Microsoft Edge (xem README).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
REPO_URL = (ROOT / "report" / "repo_url.txt").read_text(encoding="utf-8").strip()


def f(x, nd=4):
    """Định dạng số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm ngăn nghìn."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m.replace('.', ',')} × 10<sup>{str(int(e)).replace('-', '−')}</sup>"


def g_(x):
    return f"{x:g}".replace(".", ",")


def fig(name, caption):
    return f'<figure><img src="../outputs/figures/{name}"></figure><div class="cap">{caption}</div>'


d, g, eq, tm = R["data"], R["numpy_gd"], R["equivalence"], R["test_metrics"]
cm, er, nt, un, cv, ci, cal = R["confusion"], R["errors"], R["newton"], R["unscaled"], R["cv_C"], R["bootstrap_ci"], R["calibration"]
NP, SK, BL = "NumPy (gradient descent)", "sklearn LogisticRegression", "Baseline (prior)"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Log loss"]

metric_rows = "".join(
    f"<tr><td>{name}</td>" + "".join(f"<td>{f(tm[key][m])}</td>" for m in METRICS) + "</tr>"
    for name, key in (("Baseline (theo tỷ lệ lớp)", BL), ("NumPy, gradient descent", NP), ("sklearn LogisticRegression", SK)))
coef_rows = "".join(
    f"<tr><td>{n}</td><td>{f(a, 6)}</td><td>{f(b, 6)}</td><td>{sci(abs(a - b))}</td></tr>"
    for n, a, b in zip(R["coef"]["features"] + ["bias b"],
                       R["coef"]["w_numpy"] + [R["coef"]["b_numpy"]],
                       R["coef"]["w_sklearn"] + [R["coef"]["b_sklearn"]]))
gm = er["group_means"]
grp_rows = "".join(
    f"<tr><td>{k} — {lab}</td><td>{int(gm[k]['số mẫu'])}</td><td>{f(gm[k]['Area'], 0)}</td>"
    f"<td>{f(gm[k]['Major_Axis_Length'], 1)}</td><td>{f(gm[k]['Minor_Axis_Length'], 1)}</td><td>{f(gm[k]['p'], 3)}</td></tr>"
    for k, lab in (("TN", "Osmancik đoán đúng"), ("FP", "Osmancik đoán thành Cammeo"),
                   ("FN", "Cammeo đoán thành Osmancik"), ("TP", "Cammeo đoán đúng")))
lr_rows = "".join(
    f"<tr><td>{f(r['eta'], 3)}</td><td>{f(r['log loss sau 300 vòng'])}</td><td>{r['số vòng loss tăng']}</td></tr>"
    for r in R["lr_sweep"])
cv_rows = "".join(f"<tr><td>{g_(r['C'])}</td><td>{f(r['CV log loss (mean)'])}</td><td>{f(r['SE'])}</td></tr>" for r in cv["table"])
ci_rows = "".join(f"<tr><td>{m}</td><td>{f(ci[m]['Giá trị'])}</td><td>[{f(ci[m]['CI 95% dưới'], 3)}; {f(ci[m]['CI 95% trên'], 3)}]</td></tr>" for m in METRICS)
sgd_rows = "".join(f"<tr><td>{r['Phương pháp']}</td><td>{f(r['eta'], 3)}</td><td>{f(r['số bước cập nhật'], 0)}</td><td>{f(r['log loss train cuối'])}</td></tr>" for r in R["sgd"])
pct_c, pct_o = 100 * d["n_cammeo"] / d["n"], 100 * d["n_osmancik"] / d["n"]
err_pct = 100 * er["n_wrong"] / er["n_test"]
tb = cal["test_at_tau_best"]

html = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Bài tập 2 — Logistic Regression — B26CHKH096</title>
<link rel="stylesheet" href="report.css"></head><body>

<div class="cover">
  <div class="org">Học viện Công nghệ Bưu chính Viễn thông<br>Khoa Đào tạo Sau đại học</div>
  <div class="rule"></div>
  <div class="kind">Báo cáo bài tập 2</div>
  <div class="course">Môn học: Nhận dạng mẫu</div>
  <div class="title">Cài đặt và đánh giá Logistic Regression bằng NumPy, so sánh với scikit-learn trên bộ dữ liệu Rice (Cammeo và Osmancik)</div>
  <table class="who">
    <tr><td>Giảng viên:</td><td><b>TS. Nguyễn Xuân Đức</b></td></tr>
    <tr><td>Học viên:</td><td><b>Trần Vũ Trường</b></td></tr>
    <tr><td>Mã học viên:</td><td><b>B26CHKH096</b></td></tr>
    <tr><td>Lớp:</td><td><b>M26CQKH02-B</b></td></tr>
  </table>
  <div class="place">Hà Nội, tháng 10 năm 2026</div>
</div>

<h1>Tóm tắt</h1>
<p>Báo cáo cài đặt Logistic Regression nhị phân bằng NumPy (score, sigmoid, log loss, gradient descent cập nhật trọng số và bias) và so sánh với <code>sklearn.linear_model.LogisticRegression</code> trên bộ dữ liệu Rice ({f(d['n'], 0)} hạt gạo, {d['n_features']} đặc trưng hình thái, 2 giống). Hai bản được cho tối ưu cùng một hàm mục tiêu nên cho kết quả trùng nhau trên tập Test: accuracy {f(tm[NP]['Accuracy'])}, F1 {f(tm[NP]['F1'])}, ROC-AUC {f(tm[NP]['ROC-AUC'])}, log loss {f(tm[NP]['Log loss'])}; xác suất dự đoán lệch nhau tối đa {sci(eq['max_dp_test'])}. Khác biệt nằm ở tốc độ: gradient descent cần {f(g['n_iter'], 0)} vòng lặp, bộ giải lbfgs của sklearn cần {eq['sk_n_iter']} vòng, phương pháp Newton tự cài cần {nt['n_iter']} vòng. Nguyên nhân là các đặc trưng kích thước gần cộng tuyến. Lỗi phân loại tập trung ở các hạt có kích thước trung gian giữa hai giống.</p>

<h1>1. Bài toán và mục tiêu</h1>
<p><b>Đối tượng:</b> một hạt gạo được chụp ảnh. <b>Đầu vào:</b> 7 đặc trưng hình thái trích từ ảnh. <b>Đầu ra:</b> giống gạo, Cammeo hoặc Osmancik. Quy ước lớp dương y = 1 là Cammeo; precision, recall và F1 trong báo cáo đều tính cho lớp này.</p>
<p><b>Mục tiêu:</b> (1) tự cài Logistic Regression bằng NumPy, không dùng mô hình hay bộ tối ưu có sẵn; (2) huấn luyện <code>LogisticRegression</code> của scikit-learn trên cùng dữ liệu đã tiền xử lý; (3) so sánh hai bản bằng accuracy, precision, recall, F1, ROC-AUC, log loss và ma trận nhầm lẫn; (4) nhận xét về sự hội tụ, chênh lệch giữa hai bản và các lỗi phân loại thường gặp.</p>

<h1>2. Dữ liệu</h1>
<p>Bộ dữ liệu Rice (Cammeo and Osmancik) lấy từ UCI Machine Learning Repository (id 545), do Cinar và Koklu công bố năm 2019. Dữ liệu gồm {f(d['n'], 0)} mẫu, không có ô thiếu và không có dòng trùng lặp.</p>
<table>
<tr><th>Đặc trưng</th><th>Ý nghĩa</th></tr>
<tr><td>Area</td><td class="l">Số pixel bên trong đường biên hạt gạo</td></tr>
<tr><td>Perimeter</td><td class="l">Chu vi đường biên</td></tr>
<tr><td>Major_Axis_Length</td><td class="l">Độ dài trục chính (đoạn dài nhất vẽ được trên hạt)</td></tr>
<tr><td>Minor_Axis_Length</td><td class="l">Độ dài trục phụ</td></tr>
<tr><td>Eccentricity</td><td class="l">Độ lệch tâm của elip tương đương (gần 1 là dẹt dài)</td></tr>
<tr><td>Convex_Area</td><td class="l">Số pixel của bao lồi nhỏ nhất chứa hạt</td></tr>
<tr><td>Extent</td><td class="l">Tỷ lệ diện tích hạt trên diện tích hình chữ nhật bao quanh</td></tr>
</table>
<div class="cap">Bảng 1. Bảy đặc trưng của bộ dữ liệu Rice.</div>
<p><b>Nhãn và tỷ lệ lớp:</b> Osmancik {f(d['n_osmancik'], 0)} mẫu ({f(pct_o, 1)}%), Cammeo {f(d['n_cammeo'], 0)} mẫu ({f(pct_c, 1)}%). Hai lớp lệch nhẹ; mô hình luôn đoán lớp đông hơn sẽ đạt accuracy khoảng {f(pct_o, 0)}%, nên accuracy vẫn dùng được nhưng cần đọc kèm precision và recall.</p>
<p><b>Cộng tuyến:</b> các đặc trưng kích thước tương quan rất mạnh với nhau (Area và Convex_Area: 0,999; Area và Perimeter: 0,966). Đặc điểm này không làm giảm độ chính xác nhưng quyết định tốc độ hội tụ của gradient descent (mục 6.1).</p>
{fig("01_lop_va_tuong_quan.png", "Hình 1. Tỷ lệ hai lớp và ma trận tương quan giữa các đặc trưng.")}

<h1>3. Biểu diễn và tiền xử lý</h1>
<ul>
<li><b>Chia dữ liệu:</b> phân tầng theo nhãn thành Train {f(d['n_train'], 0)} mẫu (60%), Validation {d['n_val']} mẫu (20%) và Test {d['n_test']} mẫu (20%), hạt giống ngẫu nhiên cố định. Train dùng để ước lượng tham số, Validation để chọn ngưỡng, Test chỉ dùng cho báo cáo cuối (Slide 1, tr. 19).</li>
<li><b>Chuẩn hoá:</b> z = (x − μ) / σ, với μ và σ học trên Train rồi dùng lại cho Validation và Test (Slide 5, tr. 21). Làm vậy để không rò rỉ thông tin từ Test vào bước tiền xử lý (Slide 3, tr. 14).</li>
<li><b>Cả hai bản dùng chung</b> ma trận X<sub>train</sub> đã chuẩn hoá này.</li>
</ul>

<h1>4. Phương pháp</h1>
<h2>4.1. Cài đặt bằng NumPy</h2>
<p>Mã nguồn nằm trong <code>src/logreg_numpy.py</code>, chỉ dùng phép toán mảng. Ký hiệu theo Slide 6 (tr. 10–22):</p>
<div class="eq">s = w<sup>T</sup>x + b, &nbsp;&nbsp; p = 1 / (1 + e<sup>−s</sup>)</div>
<div class="eq">L = (1/n) Σ<sub>i</sub> −[ y<sub>i</sub> ln p<sub>i</sub> + (1 − y<sub>i</sub>) ln(1 − p<sub>i</sub>) ], &nbsp;&nbsp; J = L + λ Σ<sub>j</sub> w<sub>j</sub><sup>2</sup></div>
<div class="eq">∇<sub>w</sub> J = X<sup>T</sup>(p − y) / n + 2λw, &nbsp;&nbsp; ∂J/∂b = mean(p − y)</div>
<div class="eq">w ← w − η ∇<sub>w</sub> J, &nbsp;&nbsp; b ← b − η ∂J/∂b</div>
<ul>
<li><b>Ổn định số:</b> sigmoid tách hai nhánh s ≥ 0 và s &lt; 0 để không gọi hàm mũ với số dương lớn; log loss tính trực tiếp từ score bằng ln(1 + e<sup>s</sup>) − y·s nên không gặp ln(0).</li>
<li><b>Regularization:</b> sklearn tối thiểu hoá ½‖w‖² + C Σ ℓ<sub>i</sub>. Chia cho C·n được đúng dạng J ở trên với λ = 1/(2Cn). Với C = 1 và n = {f(d['n_train'], 0)}, λ = {sci(g['lambda'])}. Bias không bị phạt. C điều khiển regularization, không phải bước học (Slide 6, tr. 21–22).</li>
<li><b>Learning rate:</b> η = 1/L<sub>max</sub> = {f(g['eta'], 3)}, trong đó L<sub>max</sub> = λ<sub>max</sub>(X̃<sup>T</sup>X̃)/(4n) + 2λ = {f(g['L_max'], 3)} là cận trên độ cong của J (do p(1 − p) ≤ 1/4). Với bước này J chắc chắn giảm ở mọi vòng lặp.</li>
<li><b>Khởi tạo:</b> w = 0, b = 0. <b>Cập nhật đồng thời</b> w và b từ cùng một gradient (Slide 5, tr. 11–12).</li>
<li><b>Điều kiện dừng:</b> ‖∇J‖<sub>2</sub> &lt; 10<sup>−6</sup>, hoặc tối đa {f(g['max_iter'], 0)} vòng lặp.</li>
</ul>
<h2>4.2. Kiểm thử cài đặt</h2>
<ul>
<li><b>Ví dụ tính tay trong slide</b> (Slide 6, tr. 14 và tr. 21): với x = (1; 2), w = (0,8; −0,3), b = −1, y = 1, η = 0,1, mã nguồn cho s = −0,8; p = 0,310; ℓ = 1,17; sau một bước w = (0,869; −0,162), b = −0,931, p = 0,405, ℓ = 0,90. Khớp với slide.</li>
<li><b>Kiểm tra gradient:</b> sai số tương đối giữa gradient giải tích và sai phân hữu hạn trung tâm là {sci(R['grad_check_rel_err'])}.</li>
</ul>
<h2>4.3. Scikit-learn</h2>
<p><code>LogisticRegression(C=1.0, tol=1e-10, max_iter=10000)</code>, phạt L2, bộ giải lbfgs, huấn luyện trên cùng X<sub>train</sub> đã chuẩn hoá. Đặt <code>tol</code> rất nhỏ để lbfgs chạy tới nghiệm chính xác; nhờ đó chênh lệch còn lại giữa hai bản, nếu có, đến từ thuật toán tối ưu chứ không do dừng sớm.</p>
<h2>4.4. Baseline</h2>
<p><code>DummyClassifier(strategy="prior")</code>: luôn dự đoán xác suất bằng tỷ lệ lớp trên Train (Slide 6, tr. 8–9).</p>

<h1>5. Thực nghiệm và kết quả</h1>
<h2>5.1. Quá trình huấn luyện</h2>
<p>Gradient descent dừng theo điều kiện ‖∇J‖ &lt; 10<sup>−6</sup> sau {f(g['n_iter'], 0)} vòng lặp. Log loss trên Train giảm từ 0,6931 xuống {f(g['train_log_loss'])}: sau 10 vòng còn {f(g['loss_at']['10'])}, sau 100 vòng còn {f(g['loss_at']['100'])}, sau 1.000 vòng còn {f(g['loss_at']['1000'])}.</p>
{fig("02_log_loss_train.png", "Hình 2. Log loss trên Train và chuẩn gradient theo vòng lặp (bản NumPy).")}
<h2>5.2. Bảng so sánh trên tập Test</h2>
<p>Ngưỡng quyết định τ = 0,5. ROC-AUC và log loss tính từ xác suất dự đoán; các chỉ số còn lại tính từ nhãn dự đoán.</p>
<table>
<tr><th>Mô hình</th>{"".join(f"<th>{m}</th>" for m in METRICS)}</tr>
{metric_rows}
</table>
<div class="cap">Bảng 2. Kết quả trên tập Test ({d['n_test']} mẫu).</div>
<h2>5.3. Ma trận nhầm lẫn</h2>
<table>
<tr><th rowspan="2">Thực tế</th><th colspan="2">NumPy — dự đoán</th><th colspan="2">sklearn — dự đoán</th></tr>
<tr><th>Osmancik (0)</th><th>Cammeo (1)</th><th>Osmancik (0)</th><th>Cammeo (1)</th></tr>
<tr><td>Osmancik (0)</td><td>TN = {cm['numpy'][0][0]}</td><td>FP = {cm['numpy'][0][1]}</td><td>TN = {cm['sklearn'][0][0]}</td><td>FP = {cm['sklearn'][0][1]}</td></tr>
<tr><td>Cammeo (1)</td><td>FN = {cm['numpy'][1][0]}</td><td>TP = {cm['numpy'][1][1]}</td><td>FN = {cm['sklearn'][1][0]}</td><td>TP = {cm['sklearn'][1][1]}</td></tr>
</table>
<div class="cap">Bảng 3. Ma trận nhầm lẫn của hai bản trên Test. Số mẫu hai bản dự đoán khác nhau: {cm['n_disagree']}.</div>
{fig("04_confusion_roc.png", "Hình 3. Ma trận nhầm lẫn của hai bản và đường cong ROC trên Test.")}
<h2>5.4. Hai bản có cho cùng một mô hình không?</h2>
<table>
<tr><th>Tham số</th><th>NumPy</th><th>sklearn</th><th>|chênh lệch|</th></tr>
{coef_rows}
</table>
<div class="cap">Bảng 4. Trọng số (trên đặc trưng đã chuẩn hoá) và bias của hai bản.</div>
<p>Hàm mục tiêu J trên Train: NumPy {f(eq['J_numpy'], 9)}, sklearn {f(eq['J_sklearn'], 9)}, chênh {sci(eq['J_numpy'] - eq['J_sklearn'])}. Xác suất dự đoán trên Test lệch tối đa {sci(eq['max_dp_test'])}.</p>

<h1>6. Nhận xét</h1>
<h2>6.1. Sự hội tụ</h2>
<p>Với η = 1/L<sub>max</sub>, log loss giảm đơn điệu, không có vòng lặp nào loss tăng. Phần lớn mức giảm diễn ra trong 100 vòng đầu, nhưng phải tới vòng {f(g['n_iter'], 0)} chuẩn gradient mới xuống dưới 10<sup>−6</sup>. Đuôi hội tụ dài này do cộng tuyến: Hessian tại nghiệm có trị riêng lớn nhất {f(nt['eig_max'], 3)} và nhỏ nhất {sci(nt['eig_min'])}, số điều kiện khoảng {f(nt['kappa'], 0)}. Mặt loss có hướng rất phẳng, gradient descent đi rất chậm dọc hướng đó.</p>
<p>Thí nghiệm learning rate (300 vòng mỗi giá trị) cho đủ các dạng đường loss mô tả ở Slide 5, tr. 14 và tr. 18: η = 0,01 giảm quá chậm; η từ 0,87 đến 8 giảm đều; η = 20 dao động quanh nghiệm; η = 60 nhảy liên tục và không hội tụ. Độ cong thật lớn nhất ({f(nt['eig_max'], 3)}) nhỏ hơn nhiều so với cận L<sub>max</sub> = {f(g['L_max'], 3)}, nên η = 1/L<sub>max</sub> an toàn nhưng dè dặt, và ngưỡng mất ổn định thực tế nằm quanh 2/{f(nt['eig_max'], 3)} ≈ {f(2 / nt['eig_max'], 0)}.</p>
<table>
<tr><th>η</th><th>Log loss Train sau 300 vòng</th><th>Số vòng loss tăng</th></tr>
{lr_rows}
</table>
<div class="cap">Bảng 5. Ảnh hưởng của learning rate.</div>
{fig("03_learning_rate.png", "Hình 4. Đường loss với các learning rate khác nhau.")}
<h2>6.2. Chênh lệch giữa hai bản</h2>
<ul>
<li>Vì tối ưu cùng một hàm mục tiêu, hai bản cho {d['n_test']}/{d['n_test']} dự đoán giống nhau trên Test, hai ma trận nhầm lẫn giống hệt, accuracy, precision, recall, F1 và ROC-AUC bằng nhau; log loss chỉ khác ở chữ số thập phân thứ sáu ({f(tm[NP]['Log loss'], 6)} và {f(tm[SK]['Log loss'], 6)}).</li>
<li>Trọng số lệch nhiều hơn xác suất: tối đa {sci(eq['max_dw'])}, ở Area và Convex_Area, hai đặc trưng tương quan 0,999. Dọc hướng phẳng của mặt loss, trọng số có thể dịch chuyển mà dự đoán hầu như không đổi. Mức lệch này khớp với ước lượng ‖∇J‖ / λ<sub>min</sub> ≈ 10<sup>−6</sup> / {sci(nt['eig_min'])} ≈ {sci(1e-6 / nt['eig_min'])}.</li>
<li>Khác biệt thật sự là thuật toán: lbfgs dùng thông tin độ cong nên dừng sau {eq['sk_n_iter']} vòng; gradient descent chỉ dùng gradient nên cần {f(g['n_iter'], 0)} vòng.</li>
<li>Nếu bản NumPy không có L2 trong khi sklearn dùng C = 1 mặc định, hai bản sẽ tối ưu hai hàm khác nhau và chênh lệch quan sát được sẽ là chênh lệch về mô hình chứ không phải về cài đặt.</li>
</ul>
<h2>6.3. Các lỗi phân loại thường gặp</h2>
<p>Mô hình sai {er['n_wrong']}/{er['n_test']} mẫu Test ({f(err_pct, 1)}%): {er['fp']} FP (Osmancik bị đoán là Cammeo) và {er['fn']} FN (Cammeo bị đoán là Osmancik).</p>
<table>
<tr><th>Nhóm</th><th>Số mẫu</th><th>Area</th><th>Trục chính</th><th>Trục phụ</th><th>p trung bình</th></tr>
{grp_rows}
</table>
<div class="cap">Bảng 6. Giá trị trung bình của một số đặc trưng theo nhóm kết quả trên Test.</div>
<ul>
<li><b>Sai ở đâu:</b> ở vùng giao giữa hai giống theo kích thước hạt. FP là các hạt Osmancik to và dài bất thường (Area trung bình {f(gm['FP']['Area'], 0)}, trong khi Osmancik đoán đúng là {f(gm['TN']['Area'], 0)}). FN là các hạt Cammeo nhỏ và ngắn (Area {f(gm['FN']['Area'], 0)}, so với {f(gm['TP']['Area'], 0)} của Cammeo đoán đúng).</li>
<li><b>Mức độ tự tin:</b> {er['near_threshold']}/{er['n_wrong']} mẫu sai ({f(100 * er['near_threshold'] / er['n_wrong'], 0)}%) có xác suất trong khoảng (0,3; 0,7), tức nằm sát ngưỡng (Slide 6, tr. 49); chỉ {er['confident_wrong']} mẫu sai mà mô hình rất tự tin (p &lt; 0,1 hoặc p &gt; 0,9).</li>
<li><b>Vì sao:</b> nguyên nhân thuộc nhóm đặc trưng (nhóm F trong Slide 1, tr. 22) chứ không phải mô hình. Cả 7 đặc trưng đều mô tả kích thước và hình dạng, nên một hạt Osmancik to không phân biệt được với một hạt Cammeo nhỏ.</li>
<li><b>Sửa thế nào:</b> bổ sung đặc trưng khác loại (màu sắc, kết cấu bề mặt). Đổi thuật toán tối ưu không giúp gì, vì cả ba thuật toán đã thử đều tới cùng một nghiệm.</li>
</ul>
{fig("05_loi_phan_loai.png", "Hình 5. Phân bố xác suất dự đoán trên Test và vị trí các mẫu sai theo hai trục của hạt.")}

<h1>7. Phân tích mở rộng</h1>
<h2>7.1. Newton (IRLS) và vai trò của chuẩn hoá</h2>
<p>Phương pháp Newton tự cài bằng NumPy, cập nhật θ ← θ − H<sup>−1</sup>∇J, hội tụ sau {nt['n_iter']} vòng với cùng ngưỡng dừng, vì không bị ảnh hưởng bởi số điều kiện. Ngược lại, khi <b>không</b> chuẩn hoá đặc trưng, L<sub>max</sub> tăng lên {sci(un['L_max'])} nên bước học an toàn chỉ còn {sci(un['eta'])}; sau {f(un['n_iter'], 0)} vòng log loss mới giảm tới {f(un['train_log_loss'], 3)} và accuracy Test là {f(un['test_accuracy'], 3)} (Slide 5, tr. 17).</p>
{fig("06_gd_vs_newton.png", "Hình 6. Chuẩn gradient theo vòng lặp: gradient descent và Newton.")}
<h2>7.2. Batch, mini-batch và SGD</h2>
<table>
<tr><th>Phương pháp</th><th>η</th><th>Số bước cập nhật</th><th>Log loss Train sau 30 epoch</th></tr>
{sgd_rows}
</table>
<div class="cap">Bảng 7. So sánh với cùng 30 lượt duyệt dữ liệu (Slide 5, tr. 16).</div>
<p>Tính theo số lượt duyệt dữ liệu, mini-batch và SGD xuống gần nghiệm nhanh hơn batch; đổi lại đường loss của SGD dao động quanh nghiệm thay vì giảm đều.</p>
<h2>7.3. Chọn C bằng cross-validation</h2>
<table>
<tr><th>C</th><th>CV log loss (5 fold)</th><th>SE</th></tr>
{cv_rows}
</table>
<div class="cap">Bảng 8. Chọn C trên Train, scaler nằm trong pipeline và được fit lại ở mỗi fold (Slide 6, tr. 24 và tr. 45).</div>
<p>CV log loss thấp nhất ở C = {cv['C_best']:g}; theo quy tắc một sai số chuẩn (Slide 5, tr. 45), C = {cv['C_one_se']:g} là mô hình đơn giản nhất còn trong ngưỡng. Trên Test, C = {cv['C_best']:g} cho accuracy {f(cv['test_metrics_best']['Accuracy'])} và log loss {f(cv['test_metrics_best']['Log loss'])}, so với {f(tm[SK]['Accuracy'])} và {f(tm[SK]['Log loss'])} của C = 1. Mức cải thiện này nhỏ hơn bề rộng khoảng tin cậy ở Bảng 9.</p>
<h2>7.4. Độ bất định, chất lượng xác suất và ngưỡng</h2>
<table>
<tr><th>Chỉ số</th><th>Giá trị trên Test</th><th>Khoảng tin cậy 95% (bootstrap)</th></tr>
{ci_rows}
</table>
<div class="cap">Bảng 9. Khoảng tin cậy từ 2.000 lần bootstrap trên Test (bản NumPy).</div>
<ul>
<li>Hai bản không có cặp dự đoán bất đồng nào (kiểm định McNemar: {R['mcnemar']['n01']} và {R['mcnemar']['n10']}), nên không có khác biệt để kiểm định.</li>
<li>Brier score trên Test là {f(cal['brier'])}; đường calibration nằm gần đường chéo, lệch nhiều nhất ở nhóm xác suất quanh 0,5 (Slide 6, tr. 40–42).</li>
<li>Quét ngưỡng trên Validation theo F1 cho τ = {f(cal['tau_best_val'], 2)}. Áp dụng lên Test, F1 chỉ đổi từ {f(tm[NP]['F1'])} lên {f(tb['F1'])} (precision {f(tb['Precision'], 3)}, recall {f(tb['Recall'], 3)}), nên giữ τ = 0,5 là hợp lý khi hai loại lỗi có chi phí như nhau (Slide 6, tr. 34–35).</li>
<li>VIF của Area và Convex_Area trên 1.000, nên odds ratio của từng đặc trưng kích thước riêng lẻ không ổn định và không nên diễn giải tách rời (Slide 6, tr. 26–28).</li>
</ul>
{fig("09_calibration_nguong.png", "Hình 7. Calibration trên Test và F1 theo ngưỡng trên Validation.")}

<h1>8. Kết luận</h1>
<ul>
<li>Bản NumPy cài đúng: tái lập được ví dụ tính tay của slide, qua kiểm tra gradient, và cho cùng nghiệm với scikit-learn khi hai bản tối ưu cùng một hàm mục tiêu.</li>
<li>Trên bộ Rice, Logistic Regression đạt accuracy {f(tm[NP]['Accuracy'], 3)} (khoảng tin cậy 95%: [{f(ci['Accuracy']['CI 95% dưới'], 3)}; {f(ci['Accuracy']['CI 95% trên'], 3)}]) và ROC-AUC {f(tm[NP]['ROC-AUC'], 3)}, so với accuracy {f(tm[BL]['Accuracy'], 3)} của baseline.</li>
<li>Gradient descent hội tụ chắc chắn với η = 1/L<sub>max</sub> nhưng chậm khi đặc trưng cộng tuyến; chuẩn hoá đặc trưng là điều kiện để nó chạy được trong thực tế.</li>
<li><b>Giới hạn:</b> mô hình chỉ dùng 7 đặc trưng hình thái nên không tách được các hạt có kích thước trung gian; kết quả áp dụng cho ảnh chụp trong cùng điều kiện với bộ dữ liệu gốc.</li>
</ul>

<h1>9. Mã nguồn và cách chạy</h1>
<p>Mã nguồn công khai: <a href="{REPO_URL}">{REPO_URL}</a></p>
<pre>pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace B26CHKH096_TranVuTruong_BaiTap2.ipynb</pre>
<p>Hoặc mở notebook và chọn <i>Restart &amp; Run All</i>. Dữ liệu nằm sẵn trong <code>data/</code>, không cần tải thêm. Môi trường đã chạy: Python 3.12, NumPy 2.5.3, scikit-learn 1.9.1.</p>

<h1>Tài liệu tham khảo</h1>
<ol>
<li>Nguyễn Xuân Đức. Bài giảng môn Nhận dạng mẫu, Học viện Công nghệ Bưu chính Viễn thông: Slide 1 “Tổng quan”; Slide 3 “Dữ liệu &amp; đặc trưng”; Slide 5 “Mô hình hồi quy”; Slide 6 “Hồi quy logistic”. Số trang trích dẫn là số thứ tự trang trong file PDF.</li>
<li>C. M. Bishop. <i>Pattern Recognition and Machine Learning</i>. Springer, 2006. Mục 4.3 (Probabilistic Discriminative Models).</li>
<li>I. Cinar, M. Koklu. “Classification of Rice Varieties Using Artificial Intelligence Methods.” <i>International Journal of Intelligent Systems and Applications in Engineering</i>, 7(3), 2019, 188–194.</li>
<li>Rice (Cammeo and Osmancik). UCI Machine Learning Repository, 2019. https://archive.ics.uci.edu/dataset/545</li>
<li>F. Pedregosa và cộng sự. “Scikit-learn: Machine Learning in Python.” <i>Journal of Machine Learning Research</i>, 12, 2011, 2825–2830.</li>
</ol>
</body></html>"""

(ROOT / "report" / "report.html").write_text(html, encoding="utf-8")
print("Đã ghi report/report.html")
