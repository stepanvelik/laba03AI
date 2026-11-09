import numpy as np
import matplotlib.pyplot as plt

RANGES = [(-0.7, -0.5), (0.5, 0.7)]
X_MIN, X_MAX = -1.0, 1.0
SEED = 42


def in_target_range(x: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(x, dtype=bool)
    for left, right in RANGES:
        mask |= (x >= left) & (x <= right)
    return mask


def make_dataset(n_samples: int = 4000, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    x = rng.uniform(X_MIN, X_MAX, size=(n_samples, 1))
    y = in_target_range(x).astype(np.float64)
    return x, y


def balance_classes(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Досэмплировать меньший класс до размера большего (oversampling)."""
    pos = np.where(y.ravel() == 1)[0]
    neg = np.where(y.ravel() == 0)[0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("В выборке должен быть хотя бы один пример каждого класса")

    if n_pos < n_neg:
        extra = rng.choice(pos, size=n_neg - n_pos, replace=True)
        idx = np.concatenate([neg, pos, extra])
    elif n_neg < n_pos:
        extra = rng.choice(neg, size=n_pos - n_neg, replace=True)
        idx = np.concatenate([pos, neg, extra])
    else:
        idx = np.arange(len(y))

    rng.shuffle(idx)
    return x[idx], y[idx]


def class_weights(y: np.ndarray) -> np.ndarray:
    """Веса обратно пропорциональны частоте класса (на случай остаточного дисбаланса)."""
    y_flat = y.ravel()
    n = len(y_flat)
    n_pos = np.sum(y_flat == 1)
    n_neg = n - n_pos
    w_pos = n / (2.0 * max(n_pos, 1))
    w_neg = n / (2.0 * max(n_neg, 1))
    weights = np.where(y_flat == 1, w_pos, w_neg).reshape(-1, 1)
    return weights


class MLP:
    """Сеть 1 -> 4 -> 2 -> 1, активация tanh на всех слоях."""

    def __init__(self, rng: np.random.Generator):
        self.W1 = rng.normal(0, np.sqrt(1 / 1), size=(1, 4))
        self.b1 = np.zeros((1, 4))
        self.W2 = rng.normal(0, np.sqrt(1 / 4), size=(4, 2))
        self.b2 = np.zeros((1, 2))
        self.W3 = rng.normal(0, np.sqrt(1 / 2), size=(2, 1))
        self.b3 = np.zeros((1, 1))

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)
        z3 = a2 @ self.W3 + self.b3
        a3 = np.tanh(z3)
        cache = (x, a1, a2, a3)
        return a3, cache

    def backward(self, y: np.ndarray, weights: np.ndarray, cache, lr: float, momentum: float):
        x, a1, a2, a3 = cache
        n = x.shape[0]

        da3 = (2.0 / n) * weights * (a3 - y)
        dz3 = da3 * (1.0 - a3 ** 2)
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (1.0 - a2 ** 2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1.0 - a1 ** 2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.vW3 = momentum * self.vW3 - lr * dW3
        self.vb3 = momentum * self.vb3 - lr * db3
        self.vW2 = momentum * self.vW2 - lr * dW2
        self.vb2 = momentum * self.vb2 - lr * db2
        self.vW1 = momentum * self.vW1 - lr * dW1
        self.vb1 = momentum * self.vb1 - lr * db1

        self.W3 += self.vW3
        self.b3 += self.vb3
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        a3, _ = self.forward(x)
        return a3

    def predict(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.predict_proba(x) > threshold).astype(int)


def train(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 4000,
    lr: float = 0.05,
    momentum: float = 0.9,
):
    weights = class_weights(y)
    history = []
    for epoch in range(1, epochs + 1):
        pred, cache = model.forward(x)
        loss = float(np.mean(weights * (pred - y) ** 2))
        history.append(loss)
        model.backward(y, weights, cache, lr=lr, momentum=momentum)
        if epoch % 500 == 0 or epoch == 1:
            pred_cls = (pred > 0).astype(int)
            true_cls = (y > 0).astype(int)
            acc = float(np.mean(pred_cls == true_cls))
            print(f"Эпоха {epoch:4d} | MSE={loss:.5f} | acc={acc:.4f}")
    return history


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": np.array([[tn, fp], [fn, tp]]),
    }


def plot_results(model: MLP, history: list[float], x_raw: np.ndarray, y_raw: np.ndarray):
    xs = np.linspace(X_MIN, X_MAX, 1000).reshape(-1, 1)
    true_y = in_target_range(xs).astype(float)
    net_y = model.predict_proba(xs)
    net_cls = (net_y > 0).astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(y_raw.ravel(), bins=[-0.5, 0.5, 1.5], rwidth=0.6, color="gray")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_title("Классы до балансировки")
    axes[0, 0].set_xlabel("Класс")
    axes[0, 0].set_ylabel("Число примеров")

    axes[0, 1].plot(history, color="tab:blue")
    axes[0, 1].set_title("Ошибка обучения (взвешенный MSE)")
    axes[0, 1].set_xlabel("Эпоха")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(xs, true_y, label="Истинная разметка", linewidth=2)
    axes[1, 0].plot(xs, net_y, label="Выход tanh", alpha=0.8)
    axes[1, 0].axhline(0, color="black", linewidth=0.6)
    axes[1, 0].set_title("Истина и непрерывный выход сети")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("x")

    axes[1, 1].plot(xs, true_y, label="Истина", linewidth=2)
    axes[1, 1].step(xs, net_cls, where="mid", label="Прогноз 0/1", alpha=0.8)
    axes[1, 1].set_title("Классификация (порог 0)")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("x")

    plt.tight_layout()
    plt.savefig("result.png", dpi=120)
    print("Графики сохранены в result.png")
    if plt.get_backend().lower() != "agg":
        plt.show(block=False)
        plt.pause(0.1)


def interactive_predict(model: MLP):
    print("\nВведите число x (диапазоны 1: [-0.7, -0.5] и [0.5, 0.7]).")
    print("Пустая строка или q — выход.\n")
    while True:
        raw = input("x = ").strip()
        if raw == "" or raw.lower() in {"q", "quit", "exit", "выход"}:
            print("Готово.")
            break
        raw = raw.replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            print("Нужно число, например 0.6 или -0.55")
            continue
        x = np.array([[value]], dtype=np.float64)
        score = float(model.predict_proba(x)[0, 0])
        pred = int(score > 0)
        expected = int(in_target_range(np.array([value]))[0])
        print(f"  ответ сети: {pred}  (выход tanh={score:+.4f}, по правилу диапазона: {expected})")


def main():
    rng = np.random.default_rng(SEED)

    x_raw, y_raw = make_dataset(4000, rng)
    n_pos = int(np.sum(y_raw == 1))
    n_neg = int(np.sum(y_raw == 0))
    print(f"До балансировки: класс 1 = {n_pos}, класс 0 = {n_neg} ({n_pos / len(y_raw):.1%} положительных)")

    x, y = balance_classes(x_raw, y_raw, rng)
    y_signed = np.where(y == 1, 1.0, -1.0)
    n_pos_b = int(np.sum(y == 1))
    n_neg_b = int(np.sum(y == 0))
    print(f"После oversampling: класс 1 = {n_pos_b}, класс 0 = {n_neg_b}")
    print("Архитектура: 1 -> 4 -> 2 -> 1, активация tanh")

    model = MLP(rng)
    history = train(model, x, y_signed)

    x_test, y_test = make_dataset(2000, np.random.default_rng(SEED + 1))
    y_pred = model.predict(x_test)
    stats = metrics(y_test, y_pred)
    print("\nТест (без oversampling, исходный дисбаланс):")
    print(f"  accuracy  = {stats['acc']:.4f}")
    print(f"  precision = {stats['precision']:.4f}")
    print(f"  recall    = {stats['recall']:.4f}")
    print(f"  f1        = {stats['f1']:.4f}")
    print("  confusion [[TN FP], [FN TP]]:")
    print(stats["confusion"])

    checks = np.array([[-0.6], [-0.2], [0.0], [0.6], [0.9]])
    preds = model.predict(checks).ravel()
    print("\nПроверка точек:")
    for value, pred in zip(checks.ravel(), preds):
        expected = int(in_target_range(np.array([value]))[0])
        print(f"  x={value:+.1f} -> {pred} (ожидалось {expected})")

    plot_results(model, history, x_raw, y_raw)
    interactive_predict(model)


if __name__ == "__main__":
    main()
