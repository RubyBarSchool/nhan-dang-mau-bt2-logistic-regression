"""Logistic Regression nhị phân cài bằng NumPy.

Ký hiệu theo slide "Hồi quy logistic":
    score      s = X w + b
    sigmoid    p = 1 / (1 + e^{-s})
    log loss   L = (1/n) * sum_i -[y_i ln p_i + (1 - y_i) ln(1 - p_i)]
    L2         J = L + lam * sum_j w_j^2        (không phạt bias b)
    gradient   dJ/dw = X^T (p - y) / n + 2 lam w ,  dJ/db = mean(p - y)
    cập nhật   w <- w - eta * dJ/dw ,  b <- b - eta * dJ/db

Không dùng mô hình hay bộ tối ưu có sẵn; chỉ dùng phép toán mảng của NumPy.
"""
import numpy as np


def score(X, w, b):
    return X @ w + b


def sigmoid(s):
    """Sigmoid ổn định số: không gọi exp với số dương lớn."""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    pos = s >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-s[pos]))
    e = np.exp(s[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def log_loss_from_score(s, y):
    """Log loss trung bình tính trực tiếp từ score.

    -[y ln p + (1-y) ln(1-p)] = ln(1 + e^{s}) - y s, nên không cần ln(0).
    """
    return float(np.mean(np.logaddexp(0.0, s) - y * s))


def objective(X, y, w, b, lam=0.0):
    return log_loss_from_score(score(X, w, b), y) + lam * float(w @ w)


def gradient(X, y, w, b, lam=0.0):
    p = sigmoid(score(X, w, b))
    r = p - y
    return X.T @ r / len(y) + 2.0 * lam * w, float(np.mean(r))


def lam_from_C(C, n):
    """Quy đổi C của sklearn sang lam của hàm mục tiêu J ở trên.

    sklearn tối thiểu hoá 0.5 * ||w||^2 + C * sum_i loss_i. Chia cho C * n
    được mean(loss) + ||w||^2 / (2 C n), tức lam = 1 / (2 C n).
    """
    return 1.0 / (2.0 * C * n)


def lipschitz_constant(X, lam=0.0):
    """Cận trên L của trị riêng lớn nhất của Hessian (tính cả bias).

    Hessian = Xa^T diag(p(1-p)) Xa / n + 2 lam I, với p(1-p) <= 1/4.
    """
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return float(np.linalg.eigvalsh(Xa.T @ Xa)[-1] / (4.0 * len(X)) + 2.0 * lam)


class LogisticRegressionGD:
    """Gradient descent toàn bộ batch (hoặc mini-batch nếu đặt batch_size)."""

    def __init__(self, eta=0.1, lam=0.0, max_iter=100_000, tol=1e-6,
                 batch_size=None, random_state=0):
        self.eta = eta
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y, w0=None, b0=0.0):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        w = np.zeros(d) if w0 is None else np.array(w0, dtype=float)
        b = float(b0)
        rng = np.random.default_rng(self.random_state)
        self.loss_history_ = [log_loss_from_score(score(X, w, b), y)]
        self.grad_norm_history_ = []
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            if self.batch_size is None:
                gw, gb = gradient(X, y, w, b, self.lam)
                gnorm = float(np.sqrt(gw @ gw + gb * gb))
                self.grad_norm_history_.append(gnorm)
                if gnorm < self.tol:  # điều kiện dừng: ||grad J|| < tol
                    self.converged_ = True
                    it -= 1
                    break
            else:
                idx = rng.choice(n, size=self.batch_size, replace=False)
                gw, gb = gradient(X[idx], y[idx], w, b, self.lam)
            # cập nhật đồng thời w và b từ cùng một gradient
            w = w - self.eta * gw
            b = b - self.eta * gb
            loss = log_loss_from_score(score(X, w, b), y)
            self.loss_history_.append(loss)
            if not np.isfinite(loss):
                break
        self.w_, self.b_, self.n_iter_ = w, b, it
        return self

    def decision_function(self, X):
        return score(np.asarray(X, dtype=float), self.w_, self.b_)

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict(self, X, tau=0.5):
        return (self.predict_proba(X) >= tau).astype(int)


class LogisticRegressionNewton:
    """Newton (IRLS): dùng thêm Hessian, để so tốc độ hội tụ với GD."""

    def __init__(self, lam=0.0, max_iter=100, tol=1e-6):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Xa = np.hstack([X, np.ones((n, 1))])
        theta = np.zeros(d + 1)
        reg = 2.0 * self.lam * np.r_[np.ones(d), 0.0]
        self.loss_history_ = [log_loss_from_score(Xa @ theta, y)]
        self.grad_norm_history_ = []
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            p = sigmoid(Xa @ theta)
            g = Xa.T @ (p - y) / n + reg * theta
            gnorm = float(np.linalg.norm(g))
            self.grad_norm_history_.append(gnorm)
            if gnorm < self.tol:
                self.converged_ = True
                it -= 1
                break
            H = (Xa * (p * (1 - p))[:, None]).T @ Xa / n + np.diag(reg)
            theta = theta - np.linalg.solve(H, g)
            self.loss_history_.append(log_loss_from_score(Xa @ theta, y))
        self.w_, self.b_, self.n_iter_ = theta[:d], float(theta[d]), it
        return self

    def predict_proba(self, X):
        return sigmoid(score(np.asarray(X, dtype=float), self.w_, self.b_))
