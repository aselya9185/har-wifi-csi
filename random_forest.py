import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# =========================
# PARAMETERS
# =========================
W2_values = [10, 20, 30, 40, 50]

antenna_pairs = [
    "ant01", "ant02", "ant03",
    "ant12", "ant13", "ant23"
]

base_dir = "dataset/features"
plot_root = "plots"

os.makedirs(plot_root, exist_ok=True)


# =========================
# HELPERS
# =========================
def load_dataset(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


def prepare_xy(data):
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def run_rf(X_train, y_train, X_test):
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def save_conf_matrix(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    plt.xticks([0.5,1.5,2.5,3.5], ['empty','sit','stand','walk'])
    plt.yticks([0.5,1.5,2.5,3.5], ['empty','sit','stand','walk'])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# =========================
# MAIN LOOP
# =========================
results = []

for W2 in W2_values:
    print(f"\n=== W2 = {W2} ===")

    for pair in antenna_pairs:

        print(f"Pair: {pair}")

        path_r1 = os.path.join(base_dir, f"w2_{W2}", "r1", f"r1_{pair}.npy")
        path_r2 = os.path.join(base_dir, f"w2_{W2}", "r2", f"r2_{pair}.npy")
        path_all = os.path.join(base_dir, f"w2_{W2}", "r1_r2", f"r1_r2_{pair}.npy")

        data_r1 = load_dataset(path_r1)
        data_r2 = load_dataset(path_r2)
        data_all = load_dataset(path_all)

        if data_r1 is None or data_r2 is None or data_all is None:
            continue

        # =========================
        # SETUP 1: MIXED
        # =========================
        X, y = prepare_xy(data_all)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        X_train, X_test = scale_data(X_train, X_test)

        y_pred = run_rf(X_train, y_train, X_test)

        acc = accuracy_score(y_test, y_pred)
        err = 1 - acc

        results.append(["mixed", pair, W2, acc, err])

        save_conf_matrix(
            y_test,
            y_pred,
            f"{plot_root}/confusion_matrix/w2_{W2}/mixed/{pair}.png",
            f"Mixed - {pair} - W2={W2}"
        )

        # =========================
        # SETUP 2: r1 -> r2
        # =========================
        X_train, y_train = prepare_xy(data_r1)
        X_test, y_test = prepare_xy(data_r2)

        X_train, X_test = scale_data(X_train, X_test)

        y_pred = run_rf(X_train, y_train, X_test)

        acc = accuracy_score(y_test, y_pred)
        err = 1 - acc

        results.append(["r1_to_r2", pair, W2, acc, err])

        save_conf_matrix(
            y_test,
            y_pred,
            f"{plot_root}/confusion_matrix/w2_{W2}/r1_to_r2/{pair}.png",
            f"r1→r2 - {pair} - W2={W2}"
        )

        # =========================
        # SETUP 3: r2 -> r1
        # =========================
        X_train, y_train = prepare_xy(data_r2)
        X_test, y_test = prepare_xy(data_r1)

        X_train, X_test = scale_data(X_train, X_test)

        y_pred = run_rf(X_train, y_train, X_test)

        acc = accuracy_score(y_test, y_pred)
        err = 1 - acc

        results.append(["r2_to_r1", pair, W2, acc, err])

        save_conf_matrix(
            y_test,
            y_pred,
            f"{plot_root}/confusion_matrix/w2_{W2}/r2_to_r1/{pair}.png",
            f"r2→r1-{pair}-W2={W2}"
        )


# =========================
# SAVE SUMMARY TABLE FIGURE
# =========================
df = pd.DataFrame(results, columns=[
    "setup", "antenna_pair", "W2", "accuracy", "mean_error"
])

os.makedirs(f"{plot_root}/accuracy_error", exist_ok=True)

# ---- SAVE CSV ----
csv_path = f"{plot_root}/accuracy_error/summary_table.csv"
df.to_csv(csv_path, index=False)

# ---- SAVE IMAGE ----
row_count = len(df)
fig_height = max(6, row_count * 0.3)  # dynamic scaling
fig, ax = plt.subplots(figsize=(12, fig_height))
ax.axis('off')

table = ax.table(
    cellText=df.round(3).values,
    colLabels=df.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width(col=list(range(len(df.columns))))
table.scale(1, 1.2)

plt.savefig(f"{plot_root}/accuracy_error/summary_table.png")
plt.close()

print("\nAll experiments completed and plots saved.")