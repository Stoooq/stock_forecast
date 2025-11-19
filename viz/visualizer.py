import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_preds(y_true, y_pred):
        plt.figure(figsize=(10,4))
        plt.plot(y_true, label="true")
        plt.plot(y_pred, label="pred")
        plt.legend()
        plt.title("Cena: rzeczywista vs predykcja")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_loss(history):
        plt.figure(figsize=(8,4))
        plt.plot(history["train_loss"], label="train")
        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="val")
        plt.legend()
        plt.title("Loss")
        plt.tight_layout()
        plt.show()