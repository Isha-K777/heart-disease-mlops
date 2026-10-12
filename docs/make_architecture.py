"""Generate docs/architecture.png (run: python docs/make_architecture.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)


def arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", fontsize=8, color="#444")


def group(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec="#888", lw=1, ls="--"))
    ax.text(x + 0.15, y + h - 0.3, title, fontsize=10, fontweight="bold", color="#555")


group(0.2, 4.4, 6.6, 3.3, "1. Data & experimentation")
box(0.5, 5.6, 1.7, 1.0, "UCI Heart\nDisease data", "#fde9c9")
box(2.6, 5.6, 1.8, 1.0, "download_data.py\nclean + EDA", "#dbe9f6")
box(4.8, 5.6, 1.8, 1.0, "sklearn pipeline\nLR / RF / GB\nnested CV", "#dbe9f6")
box(3.2, 4.6, 2.8, 0.7, "MLflow tracking\n(params, metrics, plots, models)", "#e3f1df")
arrow(2.2, 6.1, 2.6, 6.1); arrow(4.4, 6.1, 4.8, 6.1); arrow(5.7, 5.6, 5.0, 5.3)

group(7.2, 4.4, 6.6, 3.3, "2. CI/CD (GitHub Actions)")
box(7.5, 5.6, 1.5, 1.0, "git push\n/ PR", "#fde9c9")
box(9.3, 5.6, 1.4, 1.0, "flake8\nlint", "#e3f1df")
box(10.9, 5.6, 1.4, 1.0, "pytest\n+ coverage", "#e3f1df")
box(12.4, 5.6, 1.2, 1.0, "train\nartifacts", "#e3f1df")
box(9.3, 4.6, 4.3, 0.7, "docker build  ->  run  ->  smoke-test /predict", "#e3f1df")
arrow(9.0, 6.1, 9.3, 6.1); arrow(10.7, 6.1, 10.9, 6.1); arrow(12.3, 6.1, 12.4, 6.1)
arrow(13.0, 5.6, 12.0, 5.3)

group(0.2, 0.2, 9.0, 3.9, "3. Kubernetes (Minikube / Docker Desktop / cloud)")
box(0.5, 1.2, 1.7, 1.2, "Ingress /\nLoadBalancer\nService", "#f3d9e8")
box(2.6, 2.3, 2.2, 1.0, "Pod: FastAPI\n/predict /health", "#f6ecc8")
box(2.6, 0.8, 2.2, 1.0, "Pod: FastAPI\n(replica, HPA)", "#f6ecc8")
box(5.4, 1.5, 1.6, 1.2, "Prometheus\nscrapes\n/metrics", "#f8d9d0")
box(7.4, 1.5, 1.6, 1.2, "Grafana\ndashboards", "#f8d9d0")
arrow(2.2, 1.8, 2.6, 2.7); arrow(2.2, 1.8, 2.6, 1.3)
arrow(4.8, 2.7, 5.4, 2.2, "metrics"); arrow(4.8, 1.3, 5.4, 1.9)
arrow(7.0, 2.1, 7.4, 2.1)

box(10.0, 1.5, 1.7, 1.2, "Docker image\nheart-disease-api", "#dbe9f6")
box(12.1, 1.5, 1.6, 1.2, "Client\n(curl / app)", "#fde9c9")
arrow(10.0, 2.1, 9.2, 2.1, "deploy")
ax.plot([12.9, 12.9, 1.35], [1.5, 0.45, 0.45], color="#333", lw=1.5)
arrow(1.35, 0.45, 1.35, 1.2)
ax.text(7.0, 0.55, "HTTP JSON request / response (+ request logs)", fontsize=8, color="#444", ha="center")
arrow(11.0, 4.6, 10.85, 2.7, "image")
ax.set_title("Heart Disease MLOps - system architecture", fontsize=14, fontweight="bold")
fig.savefig("docs/architecture.png", dpi=160, bbox_inches="tight")
