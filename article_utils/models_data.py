import re
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_content: str) -> List[Tuple[int, float, float, float, float, float]]:
    pattern = re.compile(
        r'Epoch (\d+), Train Loss: ([\d.]+), Validation Accuracy: ([\d.]+), '
        r'Train Accuracy: ([\d.]+), Train Time: ([\d.]+), Eval Time: ([\d.]+)'
    )
    return [
        (int(epoch), float(loss), float(val_acc), float(train_acc), float(train_time), float(eval_time))
        for epoch, loss, val_acc, train_acc, train_time, eval_time 
        in pattern.findall(log_content)
    ]

def analyze_logs(log_content: str) -> dict:
    parsed_data = parse_log(log_content)
    
    total_epochs = len(parsed_data)
    
    first_epoch = parsed_data[0]
    last_epoch = parsed_data[-1]
    
    best_val_acc = max(parsed_data, key=lambda x: x[2])
    best_train_acc = max(parsed_data, key=lambda x: x[3])
    best_train_loss = min(parsed_data, key=lambda x: x[1])
    
    avg_train_time = sum(epoch[4] for epoch in parsed_data) / total_epochs
    avg_eval_time = sum(epoch[5] for epoch in parsed_data) / total_epochs
    
    # Calculate averages for the last 10 epochs
    last_10_epochs = parsed_data[-10:]
    avg_last_10 = {
        "train_loss": sum(epoch[1] for epoch in last_10_epochs) / 10,
        "val_accuracy": sum(epoch[2] for epoch in last_10_epochs) / 10,
        "train_accuracy": sum(epoch[3] for epoch in last_10_epochs) / 10
    }
    
    return {
        "total_epochs": total_epochs,
        "initial_metrics": {
            "train_loss": first_epoch[1],
            "val_accuracy": first_epoch[2],
            "train_accuracy": first_epoch[3]
        },
        "final_metrics": {
            "train_loss": last_epoch[1],
            "val_accuracy": last_epoch[2],
            "train_accuracy": last_epoch[3]
        },
        "best_metrics": {
            "train_loss": best_train_loss[1],
            "val_accuracy": best_val_acc[2],
            "train_accuracy": best_train_acc[3]
        },
        "avg_train_time": avg_train_time,
        "avg_eval_time": avg_eval_time,
        "avg_last_10_epochs": avg_last_10
    }

def plot_and_save_comparison(model_logs: List[Tuple[str, str]], output_prefix: str = "comparison"):
    plt.figure(figsize=(15, 5))
    metrics = ["Validation Accuracy", "Train Accuracy", "Train Loss"]
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        for name, log in model_logs:
            data = parse_log(log)
            epochs = [epoch[0] for epoch in data]
            if metric == "Validation Accuracy":
                values = [epoch[2] for epoch in data]
            elif metric == "Train Accuracy":
                values = [epoch[3] for epoch in data]
            else:  # Train Loss
                values = [epoch[1] for epoch in data]
            plt.plot(epochs, values, label=name)
        
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"../final_paper/images/{output_prefix}_comparison.png")
    plt.close()

def compare_models(model_logs: List[Tuple[str, str]]):
    plot_and_save_comparison(model_logs)
    
    print("Model Comparison Results:")
    for name, log in model_logs:
        results = analyze_logs(log)
        print(f"\n{name}:")
        print(f"  Total epochs: {results['total_epochs']}")
        print("  Final metrics:")
        for metric, value in results['final_metrics'].items():
            print(f"    {metric}: {value:.4f}")
        print("  Best metrics:")
        for metric, value in results['best_metrics'].items():
            print(f"    {metric}: {value:.4f}")
        print("  Average metrics (last 10 epochs):")
        for metric, value in results['avg_last_10_epochs'].items():
            print(f"    {metric}: {value:.4f}")
        print(f"  Average train time: {results['avg_train_time']:.2f} seconds")
        print(f"  Average eval time: {results['avg_eval_time']:.2f} seconds")

# Example usage
model1_log = """
Epoch 1, Train Loss: 0.9163, Validation Accuracy: 0.9177, Train Accuracy: 0.9073, Train Time: 72.66, Eval Time: 44.66, ETA: 2024-09-15 15:06:03
Epoch 2, Train Loss: 0.0927, Validation Accuracy: 0.9333, Train Accuracy: 0.9244, Train Time: 72.86, Eval Time: 44.83, ETA: 2024-09-15 15:06:31
Epoch 3, Train Loss: 0.0660, Validation Accuracy: 0.9433, Train Accuracy: 0.9400, Train Time: 72.51, Eval Time: 44.31, ETA: 2024-09-15 15:05:27
Epoch 4, Train Loss: 0.0562, Validation Accuracy: 0.9553, Train Accuracy: 0.9587, Train Time: 73.07, Eval Time: 44.36, ETA: 2024-09-15 15:06:10
Epoch 5, Train Loss: 0.0502, Validation Accuracy: 0.9547, Train Accuracy: 0.9574, Train Time: 72.64, Eval Time: 45.19, ETA: 2024-09-15 15:06:39
Epoch 6, Train Loss: 0.0442, Validation Accuracy: 0.9680, Train Accuracy: 0.9697, Train Time: 73.65, Eval Time: 45.08, ETA: 2024-09-15 15:07:40
Epoch 7, Train Loss: 0.0439, Validation Accuracy: 0.9690, Train Accuracy: 0.9685, Train Time: 72.79, Eval Time: 44.83, ETA: 2024-09-15 15:06:25
Epoch 8, Train Loss: 0.0379, Validation Accuracy: 0.9513, Train Accuracy: 0.9554, Train Time: 72.60, Eval Time: 44.40, ETA: 2024-09-15 15:05:44
Epoch 9, Train Loss: 0.0367, Validation Accuracy: 0.9703, Train Accuracy: 0.9707, Train Time: 72.69, Eval Time: 44.60, ETA: 2024-09-15 15:06:03
Epoch 10, Train Loss: 0.0339, Validation Accuracy: 0.9667, Train Accuracy: 0.9711, Train Time: 72.99, Eval Time: 44.70, ETA: 2024-09-15 15:06:28
Epoch 11, Train Loss: 0.0307, Validation Accuracy: 0.9743, Train Accuracy: 0.9784, Train Time: 72.95, Eval Time: 44.75, ETA: 2024-09-15 15:06:30
Epoch 12, Train Loss: 0.0313, Validation Accuracy: 0.9607, Train Accuracy: 0.9704, Train Time: 72.99, Eval Time: 45.04, ETA: 2024-09-15 15:06:51
Epoch 13, Train Loss: 0.0310, Validation Accuracy: 0.9600, Train Accuracy: 0.9711, Train Time: 73.42, Eval Time: 45.36, ETA: 2024-09-15 15:07:37
Epoch 14, Train Loss: 0.0283, Validation Accuracy: 0.9740, Train Accuracy: 0.9755, Train Time: 72.77, Eval Time: 44.70, ETA: 2024-09-15 15:06:17
Epoch 15, Train Loss: 0.0276, Validation Accuracy: 0.9777, Train Accuracy: 0.9788, Train Time: 72.63, Eval Time: 44.52, ETA: 2024-09-15 15:05:57
Epoch 16, Train Loss: 0.0267, Validation Accuracy: 0.9740, Train Accuracy: 0.9778, Train Time: 73.00, Eval Time: 44.88, ETA: 2024-09-15 15:06:41
Epoch 17, Train Loss: 0.0268, Validation Accuracy: 0.9687, Train Accuracy: 0.9778, Train Time: 72.88, Eval Time: 44.56, ETA: 2024-09-15 15:06:15
Epoch 18, Train Loss: 0.0270, Validation Accuracy: 0.9723, Train Accuracy: 0.9804, Train Time: 73.10, Eval Time: 44.91, ETA: 2024-09-15 15:06:48
Epoch 19, Train Loss: 0.0255, Validation Accuracy: 0.9637, Train Accuracy: 0.9806, Train Time: 72.79, Eval Time: 44.77, ETA: 2024-09-15 15:06:23
Epoch 20, Train Loss: 0.0236, Validation Accuracy: 0.9687, Train Accuracy: 0.9793, Train Time: 72.75, Eval Time: 44.85, ETA: 2024-09-15 15:06:25
Epoch 21, Train Loss: 0.0227, Validation Accuracy: 0.9733, Train Accuracy: 0.9837, Train Time: 73.04, Eval Time: 44.82, ETA: 2024-09-15 15:06:39
Epoch 22, Train Loss: 0.0226, Validation Accuracy: 0.9770, Train Accuracy: 0.9827, Train Time: 72.72, Eval Time: 45.04, ETA: 2024-09-15 15:06:34
Epoch 23, Train Loss: 0.0242, Validation Accuracy: 0.9670, Train Accuracy: 0.9767, Train Time: 72.94, Eval Time: 44.84, ETA: 2024-09-15 15:06:35
Epoch 24, Train Loss: 0.0230, Validation Accuracy: 0.9687, Train Accuracy: 0.9807, Train Time: 72.94, Eval Time: 44.70, ETA: 2024-09-15 15:06:28
Epoch 25, Train Loss: 0.0269, Validation Accuracy: 0.9770, Train Accuracy: 0.9824, Train Time: 73.10, Eval Time: 44.76, ETA: 2024-09-15 15:06:39
Epoch 26, Train Loss: 0.0242, Validation Accuracy: 0.9667, Train Accuracy: 0.9742, Train Time: 73.09, Eval Time: 44.60, ETA: 2024-09-15 15:06:30
Epoch 27, Train Loss: 0.0254, Validation Accuracy: 0.9667, Train Accuracy: 0.9793, Train Time: 72.88, Eval Time: 44.12, ETA: 2024-09-15 15:05:57
Epoch 28, Train Loss: 0.0241, Validation Accuracy: 0.9750, Train Accuracy: 0.9779, Train Time: 72.92, Eval Time: 46.86, ETA: 2024-09-15 15:08:08
Epoch 29, Train Loss: 0.0211, Validation Accuracy: 0.9697, Train Accuracy: 0.9831, Train Time: 73.66, Eval Time: 44.45, ETA: 2024-09-15 15:06:51
Epoch 30, Train Loss: 0.0243, Validation Accuracy: 0.9730, Train Accuracy: 0.9839, Train Time: 72.82, Eval Time: 44.33, ETA: 2024-09-15 15:06:08
Epoch 31, Train Loss: 0.0229, Validation Accuracy: 0.9727, Train Accuracy: 0.9810, Train Time: 73.84, Eval Time: 45.41, ETA: 2024-09-15 15:07:40
Epoch 32, Train Loss: 0.0224, Validation Accuracy: 0.9697, Train Accuracy: 0.9805, Train Time: 73.92, Eval Time: 45.59, ETA: 2024-09-15 15:07:51
Epoch 33, Train Loss: 0.0229, Validation Accuracy: 0.9687, Train Accuracy: 0.9789, Train Time: 74.08, Eval Time: 45.59, ETA: 2024-09-15 15:07:58
Epoch 34, Train Loss: 0.0235, Validation Accuracy: 0.9750, Train Accuracy: 0.9817, Train Time: 74.18, Eval Time: 45.76, ETA: 2024-09-15 15:08:09
Epoch 35, Train Loss: 0.0208, Validation Accuracy: 0.9720, Train Accuracy: 0.9843, Train Time: 74.41, Eval Time: 45.69, ETA: 2024-09-15 15:08:15
Epoch 36, Train Loss: 0.0242, Validation Accuracy: 0.9717, Train Accuracy: 0.9793, Train Time: 73.69, Eval Time: 44.25, ETA: 2024-09-15 15:06:51
Epoch 37, Train Loss: 0.0229, Validation Accuracy: 0.9697, Train Accuracy: 0.9813, Train Time: 72.69, Eval Time: 44.21, ETA: 2024-09-15 15:06:11
Epoch 38, Train Loss: 0.0224, Validation Accuracy: 0.9723, Train Accuracy: 0.9840, Train Time: 72.71, Eval Time: 44.22, ETA: 2024-09-15 15:06:13
Epoch 39, Train Loss: 0.0205, Validation Accuracy: 0.9743, Train Accuracy: 0.9849, Train Time: 72.91, Eval Time: 44.34, ETA: 2024-09-15 15:06:24
Epoch 40, Train Loss: 0.0221, Validation Accuracy: 0.9713, Train Accuracy: 0.9813, Train Time: 72.79, Eval Time: 44.28, ETA: 2024-09-15 15:06:18
Epoch 41, Train Loss: 0.0232, Validation Accuracy: 0.9637, Train Accuracy: 0.9809, Train Time: 72.85, Eval Time: 44.19, ETA: 2024-09-15 15:06:17
Epoch 42, Train Loss: 0.0204, Validation Accuracy: 0.9787, Train Accuracy: 0.9841, Train Time: 72.66, Eval Time: 44.16, ETA: 2024-09-15 15:06:10
Epoch 43, Train Loss: 0.0235, Validation Accuracy: 0.9720, Train Accuracy: 0.9820, Train Time: 73.08, Eval Time: 43.70, ETA: 2024-09-15 15:06:08
Epoch 44, Train Loss: 0.0240, Validation Accuracy: 0.9717, Train Accuracy: 0.9838, Train Time: 72.74, Eval Time: 44.54, ETA: 2024-09-15 15:06:24
Epoch 45, Train Loss: 0.0218, Validation Accuracy: 0.9737, Train Accuracy: 0.9842, Train Time: 75.44, Eval Time: 44.31, ETA: 2024-09-15 15:07:38
Epoch 46, Train Loss: 0.0225, Validation Accuracy: 0.9700, Train Accuracy: 0.9846, Train Time: 72.97, Eval Time: 45.66, ETA: 2024-09-15 15:07:06
Epoch 47, Train Loss: 0.0233, Validation Accuracy: 0.9547, Train Accuracy: 0.9692, Train Time: 74.15, Eval Time: 45.64, ETA: 2024-09-15 15:07:38
Epoch 48, Train Loss: 0.0208, Validation Accuracy: 0.9707, Train Accuracy: 0.9769, Train Time: 74.13, Eval Time: 45.60, ETA: 2024-09-15 15:07:36
Epoch 49, Train Loss: 0.0222, Validation Accuracy: 0.9797, Train Accuracy: 0.9846, Train Time: 74.18, Eval Time: 45.61, ETA: 2024-09-15 15:07:38
Epoch 50, Train Loss: 0.0218, Validation Accuracy: 0.9727, Train Accuracy: 0.9825, Train Time: 73.29, Eval Time: 43.91, ETA: 2024-09-15 15:06:33
Epoch 51, Train Loss: 0.0219, Validation Accuracy: 0.9747, Train Accuracy: 0.9821, Train Time: 72.81, Eval Time: 44.00, ETA: 2024-09-15 15:06:24
Epoch 52, Train Loss: 0.0215, Validation Accuracy: 0.9653, Train Accuracy: 0.9779, Train Time: 72.80, Eval Time: 43.87, ETA: 2024-09-15 15:06:21
Epoch 53, Train Loss: 0.0223, Validation Accuracy: 0.9740, Train Accuracy: 0.9825, Train Time: 72.68, Eval Time: 43.89, ETA: 2024-09-15 15:06:19
Epoch 54, Train Loss: 0.0250, Validation Accuracy: 0.9680, Train Accuracy: 0.9819, Train Time: 72.72, Eval Time: 43.96, ETA: 2024-09-15 15:06:21
Epoch 55, Train Loss: 0.0223, Validation Accuracy: 0.9697, Train Accuracy: 0.9839, Train Time: 72.84, Eval Time: 43.82, ETA: 2024-09-15 15:06:20
Epoch 56, Train Loss: 0.0217, Validation Accuracy: 0.9770, Train Accuracy: 0.9840, Train Time: 72.85, Eval Time: 44.09, ETA: 2024-09-15 15:06:26
Epoch 57, Train Loss: 0.0211, Validation Accuracy: 0.9727, Train Accuracy: 0.9808, Train Time: 72.87, Eval Time: 43.82, ETA: 2024-09-15 15:06:21
Epoch 58, Train Loss: 0.0232, Validation Accuracy: 0.9697, Train Accuracy: 0.9777, Train Time: 72.82, Eval Time: 43.92, ETA: 2024-09-15 15:06:22
Epoch 59, Train Loss: 0.0222, Validation Accuracy: 0.9743, Train Accuracy: 0.9853, Train Time: 72.69, Eval Time: 43.77, ETA: 2024-09-15 15:06:18
Epoch 60, Train Loss: 0.0221, Validation Accuracy: 0.9787, Train Accuracy: 0.9863, Train Time: 72.87, Eval Time: 43.81, ETA: 2024-09-15 15:06:21
Epoch 61, Train Loss: 0.0237, Validation Accuracy: 0.9643, Train Accuracy: 0.9740, Train Time: 72.73, Eval Time: 43.82, ETA: 2024-09-15 15:06:19
Epoch 62, Train Loss: 0.0212, Validation Accuracy: 0.9753, Train Accuracy: 0.9850, Train Time: 72.77, Eval Time: 43.87, ETA: 2024-09-15 15:06:20
Epoch 63, Train Loss: 0.0226, Validation Accuracy: 0.9650, Train Accuracy: 0.9806, Train Time: 72.78, Eval Time: 43.90, ETA: 2024-09-15 15:06:21
Epoch 64, Train Loss: 0.0242, Validation Accuracy: 0.9657, Train Accuracy: 0.9785, Train Time: 72.65, Eval Time: 43.98, ETA: 2024-09-15 15:06:20
Epoch 65, Train Loss: 0.0230, Validation Accuracy: 0.9783, Train Accuracy: 0.9844, Train Time: 72.78, Eval Time: 43.78, ETA: 2024-09-15 15:06:19
Epoch 66, Train Loss: 0.0227, Validation Accuracy: 0.9723, Train Accuracy: 0.9823, Train Time: 72.69, Eval Time: 43.81, ETA: 2024-09-15 15:06:19
Epoch 67, Train Loss: 0.0228, Validation Accuracy: 0.9693, Train Accuracy: 0.9806, Train Time: 72.78, Eval Time: 43.80, ETA: 2024-09-15 15:06:20
Epoch 68, Train Loss: 0.0227, Validation Accuracy: 0.9767, Train Accuracy: 0.9833, Train Time: 72.80, Eval Time: 43.87, ETA: 2024-09-15 15:06:20
Epoch 69, Train Loss: 0.0239, Validation Accuracy: 0.9760, Train Accuracy: 0.9846, Train Time: 72.76, Eval Time: 43.88, ETA: 2024-09-15 15:06:20
Epoch 70, Train Loss: 0.0242, Validation Accuracy: 0.9707, Train Accuracy: 0.9802, Train Time: 72.75, Eval Time: 43.86, ETA: 2024-09-15 15:06:20
Epoch 71, Train Loss: 0.0232, Validation Accuracy: 0.9730, Train Accuracy: 0.9825, Train Time: 72.78, Eval Time: 43.85, ETA: 2024-09-15 15:06:20
Epoch 72, Train Loss: 0.0233, Validation Accuracy: 0.9613, Train Accuracy: 0.9786, Train Time: 72.81, Eval Time: 43.88, ETA: 2024-09-15 15:06:20
Epoch 73, Train Loss: 0.0252, Validation Accuracy: 0.9700, Train Accuracy: 0.9815, Train Time: 72.73, Eval Time: 43.86, ETA: 2024-09-15 15:06:20
Epoch 74, Train Loss: 0.0236, Validation Accuracy: 0.9767, Train Accuracy: 0.9844, Train Time: 72.78, Eval Time: 43.90, ETA: 2024-09-15 15:06:20
Epoch 75, Train Loss: 0.0208, Validation Accuracy: 0.9757, Train Accuracy: 0.9850, Train Time: 72.72, Eval Time: 43.79, ETA: 2024-09-15 15:06:20
"""

model2_log = """
Epoch 1, Train Loss: 0.8742, Validation Accuracy: 0.9103, Train Accuracy: 0.8944, Train Time: 23.08, Eval Time: 17.54, ETA: 2024-09-15 16:59:48
Epoch 2, Train Loss: 0.0732, Validation Accuracy: 0.9457, Train Accuracy: 0.9418, Train Time: 23.06, Eval Time: 17.23, ETA: 2024-09-15 16:59:24
Epoch 3, Train Loss: 0.0538, Validation Accuracy: 0.9597, Train Accuracy: 0.9613, Train Time: 23.06, Eval Time: 17.44, ETA: 2024-09-15 16:59:39
Epoch 4, Train Loss: 0.0439, Validation Accuracy: 0.9630, Train Accuracy: 0.9647, Train Time: 23.12, Eval Time: 17.52, ETA: 2024-09-15 16:59:49
Epoch 5, Train Loss: 0.0383, Validation Accuracy: 0.9600, Train Accuracy: 0.9678, Train Time: 22.95, Eval Time: 17.45, ETA: 2024-09-15 16:59:31
Epoch 6, Train Loss: 0.0335, Validation Accuracy: 0.9700, Train Accuracy: 0.9734, Train Time: 23.08, Eval Time: 17.32, ETA: 2024-09-15 16:59:32
Epoch 7, Train Loss: 0.0302, Validation Accuracy: 0.9713, Train Accuracy: 0.9764, Train Time: 22.90, Eval Time: 17.41, ETA: 2024-09-15 16:59:26
Epoch 8, Train Loss: 0.0283, Validation Accuracy: 0.9630, Train Accuracy: 0.9752, Train Time: 22.96, Eval Time: 17.51, ETA: 2024-09-15 16:59:36
Epoch 9, Train Loss: 0.0253, Validation Accuracy: 0.9683, Train Accuracy: 0.9792, Train Time: 22.94, Eval Time: 17.45, ETA: 2024-09-15 16:59:31
Epoch 10, Train Loss: 0.0243, Validation Accuracy: 0.9657, Train Accuracy: 0.9743, Train Time: 23.05, Eval Time: 17.42, ETA: 2024-09-15 16:59:36
Epoch 11, Train Loss: 0.0224, Validation Accuracy: 0.9750, Train Accuracy: 0.9783, Train Time: 22.93, Eval Time: 17.31, ETA: 2024-09-15 16:59:22
Epoch 12, Train Loss: 0.0205, Validation Accuracy: 0.9687, Train Accuracy: 0.9752, Train Time: 22.94, Eval Time: 17.30, ETA: 2024-09-15 16:59:22
Epoch 13, Train Loss: 0.0198, Validation Accuracy: 0.9757, Train Accuracy: 0.9828, Train Time: 22.96, Eval Time: 17.56, ETA: 2024-09-15 16:59:39
Epoch 14, Train Loss: 0.0187, Validation Accuracy: 0.9713, Train Accuracy: 0.9835, Train Time: 22.91, Eval Time: 17.20, ETA: 2024-09-15 16:59:14
Epoch 15, Train Loss: 0.0181, Validation Accuracy: 0.9753, Train Accuracy: 0.9815, Train Time: 22.96, Eval Time: 17.39, ETA: 2024-09-15 16:59:28
Epoch 16, Train Loss: 0.0180, Validation Accuracy: 0.9700, Train Accuracy: 0.9844, Train Time: 23.02, Eval Time: 17.32, ETA: 2024-09-15 16:59:28
Epoch 17, Train Loss: 0.0160, Validation Accuracy: 0.9733, Train Accuracy: 0.9853, Train Time: 22.96, Eval Time: 17.26, ETA: 2024-09-15 16:59:21
Epoch 18, Train Loss: 0.0173, Validation Accuracy: 0.9803, Train Accuracy: 0.9847, Train Time: 22.99, Eval Time: 17.43, ETA: 2024-09-15 16:59:33
Epoch 19, Train Loss: 0.0159, Validation Accuracy: 0.9807, Train Accuracy: 0.9865, Train Time: 22.96, Eval Time: 17.40, ETA: 2024-09-15 16:59:29
Epoch 20, Train Loss: 0.0160, Validation Accuracy: 0.9700, Train Accuracy: 0.9868, Train Time: 22.96, Eval Time: 17.45, ETA: 2024-09-15 16:59:32
Epoch 21, Train Loss: 0.0150, Validation Accuracy: 0.9787, Train Accuracy: 0.9879, Train Time: 23.14, Eval Time: 17.28, ETA: 2024-09-15 16:59:32
Epoch 22, Train Loss: 0.0148, Validation Accuracy: 0.9800, Train Accuracy: 0.9870, Train Time: 23.08, Eval Time: 18.05, ETA: 2024-09-15 17:00:10
Epoch 23, Train Loss: 0.0139, Validation Accuracy: 0.9767, Train Accuracy: 0.9882, Train Time: 23.05, Eval Time: 17.42, ETA: 2024-09-15 16:59:36
Epoch 24, Train Loss: 0.0131, Validation Accuracy: 0.9770, Train Accuracy: 0.9884, Train Time: 22.99, Eval Time: 17.43, ETA: 2024-09-15 16:59:33
Epoch 25, Train Loss: 0.0142, Validation Accuracy: 0.9680, Train Accuracy: 0.9868, Train Time: 22.99, Eval Time: 17.36, ETA: 2024-09-15 16:59:29
Epoch 26, Train Loss: 0.0128, Validation Accuracy: 0.9723, Train Accuracy: 0.9856, Train Time: 23.04, Eval Time: 17.40, ETA: 2024-09-15 16:59:34
Epoch 27, Train Loss: 0.0128, Validation Accuracy: 0.9760, Train Accuracy: 0.9893, Train Time: 22.94, Eval Time: 17.35, ETA: 2024-09-15 16:59:26
Epoch 28, Train Loss: 0.0127, Validation Accuracy: 0.9803, Train Accuracy: 0.9886, Train Time: 23.01, Eval Time: 17.52, ETA: 2024-09-15 16:59:38
Epoch 29, Train Loss: 0.0114, Validation Accuracy: 0.9767, Train Accuracy: 0.9885, Train Time: 22.90, Eval Time: 17.81, ETA: 2024-09-15 16:59:46
Epoch 30, Train Loss: 0.0124, Validation Accuracy: 0.9737, Train Accuracy: 0.9866, Train Time: 22.98, Eval Time: 17.38, ETA: 2024-09-15 16:59:30
Epoch 31, Train Loss: 0.0118, Validation Accuracy: 0.9797, Train Accuracy: 0.9901, Train Time: 23.06, Eval Time: 17.51, ETA: 2024-09-15 16:59:40
Epoch 32, Train Loss: 0.0122, Validation Accuracy: 0.9770, Train Accuracy: 0.9887, Train Time: 23.09, Eval Time: 17.61, ETA: 2024-09-15 16:59:45
Epoch 33, Train Loss: 0.0107, Validation Accuracy: 0.9720, Train Accuracy: 0.9890, Train Time: 22.97, Eval Time: 17.43, ETA: 2024-09-15 16:59:32
Epoch 34, Train Loss: 0.0115, Validation Accuracy: 0.9797, Train Accuracy: 0.9909, Train Time: 23.21, Eval Time: 17.49, ETA: 2024-09-15 16:59:45
Epoch 35, Train Loss: 0.0113, Validation Accuracy: 0.9773, Train Accuracy: 0.9904, Train Time: 23.00, Eval Time: 17.27, ETA: 2024-09-15 16:59:28
Epoch 36, Train Loss: 0.0105, Validation Accuracy: 0.9703, Train Accuracy: 0.9881, Train Time: 23.01, Eval Time: 17.34, ETA: 2024-09-15 16:59:31
Epoch 37, Train Loss: 0.0107, Validation Accuracy: 0.9777, Train Accuracy: 0.9911, Train Time: 22.97, Eval Time: 17.49, ETA: 2024-09-15 16:59:35
Epoch 38, Train Loss: 0.0104, Validation Accuracy: 0.9767, Train Accuracy: 0.9902, Train Time: 23.42, Eval Time: 17.33, ETA: 2024-09-15 16:59:46
Epoch 39, Train Loss: 0.0103, Validation Accuracy: 0.9790, Train Accuracy: 0.9910, Train Time: 23.02, Eval Time: 17.97, ETA: 2024-09-15 16:59:54
Epoch 40, Train Loss: 0.0100, Validation Accuracy: 0.9753, Train Accuracy: 0.9912, Train Time: 23.07, Eval Time: 17.31, ETA: 2024-09-15 16:59:33
Epoch 41, Train Loss: 0.0105, Validation Accuracy: 0.9770, Train Accuracy: 0.9914, Train Time: 23.00, Eval Time: 17.30, ETA: 2024-09-15 16:59:30
Epoch 42, Train Loss: 0.0093, Validation Accuracy: 0.9827, Train Accuracy: 0.9915, Train Time: 23.09, Eval Time: 17.52, ETA: 2024-09-15 16:59:40
Epoch 43, Train Loss: 0.0094, Validation Accuracy: 0.9713, Train Accuracy: 0.9882, Train Time: 23.08, Eval Time: 17.80, ETA: 2024-09-15 16:59:49
Epoch 44, Train Loss: 0.0088, Validation Accuracy: 0.9730, Train Accuracy: 0.9893, Train Time: 23.02, Eval Time: 17.61, ETA: 2024-09-15 16:59:42
Epoch 45, Train Loss: 0.0092, Validation Accuracy: 0.9783, Train Accuracy: 0.9890, Train Time: 23.27, Eval Time: 17.52, ETA: 2024-09-15 16:59:46
Epoch 46, Train Loss: 0.0094, Validation Accuracy: 0.9773, Train Accuracy: 0.9922, Train Time: 23.03, Eval Time: 17.36, ETA: 2024-09-15 16:59:35
Epoch 47, Train Loss: 0.0090, Validation Accuracy: 0.9753, Train Accuracy: 0.9900, Train Time: 23.06, Eval Time: 17.42, ETA: 2024-09-15 16:59:37
Epoch 48, Train Loss: 0.0093, Validation Accuracy: 0.9753, Train Accuracy: 0.9908, Train Time: 23.06, Eval Time: 17.39, ETA: 2024-09-15 16:59:36
Epoch 49, Train Loss: 0.0086, Validation Accuracy: 0.9727, Train Accuracy: 0.9908, Train Time: 23.04, Eval Time: 17.51, ETA: 2024-09-15 16:59:39
Epoch 50, Train Loss: 0.0090, Validation Accuracy: 0.9780, Train Accuracy: 0.9920, Train Time: 23.00, Eval Time: 17.42, ETA: 2024-09-15 16:59:36
Epoch 51, Train Loss: 0.0082, Validation Accuracy: 0.9830, Train Accuracy: 0.9912, Train Time: 23.00, Eval Time: 17.38, ETA: 2024-09-15 16:59:35
Epoch 52, Train Loss: 0.0081, Validation Accuracy: 0.9723, Train Accuracy: 0.9876, Train Time: 23.01, Eval Time: 18.47, ETA: 2024-09-15 17:00:00
Epoch 53, Train Loss: 0.0087, Validation Accuracy: 0.9787, Train Accuracy: 0.9917, Train Time: 23.39, Eval Time: 17.61, ETA: 2024-09-15 16:59:50
Epoch 54, Train Loss: 0.0082, Validation Accuracy: 0.9793, Train Accuracy: 0.9916, Train Time: 22.96, Eval Time: 17.43, ETA: 2024-09-15 16:59:37
Epoch 55, Train Loss: 0.0079, Validation Accuracy: 0.9807, Train Accuracy: 0.9932, Train Time: 23.07, Eval Time: 17.27, ETA: 2024-09-15 16:59:36
Epoch 56, Train Loss: 0.0082, Validation Accuracy: 0.9787, Train Accuracy: 0.9916, Train Time: 23.11, Eval Time: 17.58, ETA: 2024-09-15 16:59:42
Epoch 57, Train Loss: 0.0078, Validation Accuracy: 0.9780, Train Accuracy: 0.9917, Train Time: 22.96, Eval Time: 17.72, ETA: 2024-09-15 16:59:42
Epoch 58, Train Loss: 0.0078, Validation Accuracy: 0.9747, Train Accuracy: 0.9915, Train Time: 22.96, Eval Time: 17.37, ETA: 2024-09-15 16:59:36
Epoch 59, Train Loss: 0.0069, Validation Accuracy: 0.9743, Train Accuracy: 0.9918, Train Time: 22.94, Eval Time: 17.31, ETA: 2024-09-15 16:59:35
Epoch 60, Train Loss: 0.0073, Validation Accuracy: 0.9850, Train Accuracy: 0.9907, Train Time: 23.02, Eval Time: 17.70, ETA: 2024-09-15 16:59:42
Epoch 61, Train Loss: 0.0075, Validation Accuracy: 0.9763, Train Accuracy: 0.9931, Train Time: 23.00, Eval Time: 17.38, ETA: 2024-09-15 16:59:37
Epoch 62, Train Loss: 0.0073, Validation Accuracy: 0.9843, Train Accuracy: 0.9926, Train Time: 23.07, Eval Time: 17.71, ETA: 2024-09-15 16:59:43
Epoch 63, Train Loss: 0.0076, Validation Accuracy: 0.9753, Train Accuracy: 0.9932, Train Time: 23.04, Eval Time: 17.38, ETA: 2024-09-15 16:59:38
Epoch 64, Train Loss: 0.0082, Validation Accuracy: 0.9793, Train Accuracy: 0.9939, Train Time: 23.20, Eval Time: 17.21, ETA: 2024-09-15 16:59:38
Epoch 65, Train Loss: 0.0079, Validation Accuracy: 0.9783, Train Accuracy: 0.9919, Train Time: 22.94, Eval Time: 17.24, ETA: 2024-09-15 16:59:36
Epoch 66, Train Loss: 0.0067, Validation Accuracy: 0.9830, Train Accuracy: 0.9932, Train Time: 23.01, Eval Time: 17.43, ETA: 2024-09-15 16:59:38
Epoch 67, Train Loss: 0.0072, Validation Accuracy: 0.9783, Train Accuracy: 0.9917, Train Time: 23.00, Eval Time: 17.55, ETA: 2024-09-15 16:59:39
Epoch 68, Train Loss: 0.0073, Validation Accuracy: 0.9790, Train Accuracy: 0.9940, Train Time: 23.20, Eval Time: 17.43, ETA: 2024-09-15 16:59:40
Epoch 69, Train Loss: 0.0066, Validation Accuracy: 0.9787, Train Accuracy: 0.9921, Train Time: 23.04, Eval Time: 17.79, ETA: 2024-09-15 16:59:41
Epoch 70, Train Loss: 0.0074, Validation Accuracy: 0.9757, Train Accuracy: 0.9928, Train Time: 23.02, Eval Time: 17.74, ETA: 2024-09-15 16:59:40
Epoch 71, Train Loss: 0.0069, Validation Accuracy: 0.9777, Train Accuracy: 0.9927, Train Time: 22.94, Eval Time: 17.38, ETA: 2024-09-15 16:59:39
Epoch 72, Train Loss: 0.0074, Validation Accuracy: 0.9773, Train Accuracy: 0.9936, Train Time: 22.95, Eval Time: 17.23, ETA: 2024-09-15 16:59:38
Epoch 73, Train Loss: 0.0065, Validation Accuracy: 0.9810, Train Accuracy: 0.9936, Train Time: 23.14, Eval Time: 17.45, ETA: 2024-09-15 16:59:39
Epoch 74, Train Loss: 0.0067, Validation Accuracy: 0.9820, Train Accuracy: 0.9930, Train Time: 23.06, Eval Time: 17.32, ETA: 2024-09-15 16:59:39
Epoch 75, Train Loss: 0.0063, Validation Accuracy: 0.9783, Train Accuracy: 0.9938, Train Time: 23.37, Eval Time: 17.44, ETA: 2024-09-15 16:59:39
"""

model3_log = """
Epoch 1, Train Loss: 1.2206, Validation Accuracy: 0.8607, Train Accuracy: 0.8408, Train Time: 25.65, Eval Time: 20.44, ETA: 2024-09-15 19:52:05
Epoch 2, Train Loss: 0.1248, Validation Accuracy: 0.9270, Train Accuracy: 0.9154, Train Time: 26.97, Eval Time: 19.82, ETA: 2024-09-15 19:52:56
Epoch 3, Train Loss: 0.0850, Validation Accuracy: 0.9430, Train Accuracy: 0.9374, Train Time: 24.69, Eval Time: 19.99, ETA: 2024-09-15 19:50:24
Epoch 4, Train Loss: 0.0701, Validation Accuracy: 0.9383, Train Accuracy: 0.9390, Train Time: 25.26, Eval Time: 20.09, ETA: 2024-09-15 19:51:12
Epoch 5, Train Loss: 0.0629, Validation Accuracy: 0.9560, Train Accuracy: 0.9545, Train Time: 25.24, Eval Time: 20.55, ETA: 2024-09-15 19:51:42
Epoch 6, Train Loss: 0.0543, Validation Accuracy: 0.9490, Train Accuracy: 0.9571, Train Time: 25.27, Eval Time: 20.47, ETA: 2024-09-15 19:51:39
Epoch 7, Train Loss: 0.0496, Validation Accuracy: 0.9587, Train Accuracy: 0.9553, Train Time: 27.31, Eval Time: 19.87, ETA: 2024-09-15 19:53:17
Epoch 8, Train Loss: 0.0475, Validation Accuracy: 0.9433, Train Accuracy: 0.9493, Train Time: 25.12, Eval Time: 20.18, ETA: 2024-09-15 19:51:11
Epoch 9, Train Loss: 0.0463, Validation Accuracy: 0.9573, Train Accuracy: 0.9600, Train Time: 26.17, Eval Time: 19.80, ETA: 2024-09-15 19:51:55
Epoch 10, Train Loss: 0.0424, Validation Accuracy: 0.9643, Train Accuracy: 0.9657, Train Time: 25.08, Eval Time: 19.42, ETA: 2024-09-15 19:50:20
Epoch 11, Train Loss: 0.0433, Validation Accuracy: 0.9677, Train Accuracy: 0.9677, Train Time: 24.76, Eval Time: 19.65, ETA: 2024-09-15 19:50:14
Epoch 12, Train Loss: 0.0399, Validation Accuracy: 0.9663, Train Accuracy: 0.9657, Train Time: 24.95, Eval Time: 20.23, ETA: 2024-09-15 19:51:02
Epoch 13, Train Loss: 0.0388, Validation Accuracy: 0.9637, Train Accuracy: 0.9655, Train Time: 25.18, Eval Time: 19.72, ETA: 2024-09-15 19:50:45
Epoch 14, Train Loss: 0.0371, Validation Accuracy: 0.9577, Train Accuracy: 0.9681, Train Time: 25.08, Eval Time: 19.40, ETA: 2024-09-15 19:50:20
Epoch 15, Train Loss: 0.0371, Validation Accuracy: 0.9620, Train Accuracy: 0.9686, Train Time: 25.24, Eval Time: 19.77, ETA: 2024-09-15 19:50:51
Epoch 16, Train Loss: 0.0355, Validation Accuracy: 0.9610, Train Accuracy: 0.9682, Train Time: 25.09, Eval Time: 19.47, ETA: 2024-09-15 19:50:25
Epoch 17, Train Loss: 0.0345, Validation Accuracy: 0.9687, Train Accuracy: 0.9696, Train Time: 25.03, Eval Time: 19.85, ETA: 2024-09-15 19:50:44
Epoch 18, Train Loss: 0.0353, Validation Accuracy: 0.9577, Train Accuracy: 0.9670, Train Time: 27.01, Eval Time: 19.71, ETA: 2024-09-15 19:52:29
Epoch 19, Train Loss: 0.0337, Validation Accuracy: 0.9620, Train Accuracy: 0.9689, Train Time: 25.18, Eval Time: 19.34, ETA: 2024-09-15 19:50:25
Epoch 20, Train Loss: 0.0349, Validation Accuracy: 0.9560, Train Accuracy: 0.9624, Train Time: 25.19, Eval Time: 19.66, ETA: 2024-09-15 19:50:43
Epoch 21, Train Loss: 0.0356, Validation Accuracy: 0.9580, Train Accuracy: 0.9675, Train Time: 26.73, Eval Time: 19.93, ETA: 2024-09-15 19:52:21
Epoch 22, Train Loss: 0.0348, Validation Accuracy: 0.9647, Train Accuracy: 0.9709, Train Time: 24.59, Eval Time: 19.29, ETA: 2024-09-15 19:49:54
Epoch 23, Train Loss: 0.0339, Validation Accuracy: 0.9660, Train Accuracy: 0.9713, Train Time: 25.23, Eval Time: 19.52, ETA: 2024-09-15 19:50:39
Epoch 24, Train Loss: 0.0321, Validation Accuracy: 0.9650, Train Accuracy: 0.9736, Train Time: 25.19, Eval Time: 19.66, ETA: 2024-09-15 19:50:44
Epoch 25, Train Loss: 0.0326, Validation Accuracy: 0.9670, Train Accuracy: 0.9700, Train Time: 24.88, Eval Time: 19.81, ETA: 2024-09-15 19:50:36
Epoch 26, Train Loss: 0.0354, Validation Accuracy: 0.9687, Train Accuracy: 0.9697, Train Time: 25.27, Eval Time: 19.69, ETA: 2024-09-15 19:50:49
Epoch 27, Train Loss: 0.0331, Validation Accuracy: 0.9607, Train Accuracy: 0.9668, Train Time: 25.36, Eval Time: 19.62, ETA: 2024-09-15 19:50:50
Epoch 28, Train Loss: 0.0329, Validation Accuracy: 0.9663, Train Accuracy: 0.9726, Train Time: 25.25, Eval Time: 19.54, ETA: 2024-09-15 19:50:41
Epoch 29, Train Loss: 0.0333, Validation Accuracy: 0.9637, Train Accuracy: 0.9690, Train Time: 25.04, Eval Time: 19.46, ETA: 2024-09-15 19:50:28
Epoch 30, Train Loss: 0.0307, Validation Accuracy: 0.9597, Train Accuracy: 0.9637, Train Time: 25.32, Eval Time: 20.57, ETA: 2024-09-15 19:51:30
Epoch 31, Train Loss: 0.0313, Validation Accuracy: 0.9627, Train Accuracy: 0.9701, Train Time: 24.52, Eval Time: 19.79, ETA: 2024-09-15 19:50:21
Epoch 32, Train Loss: 0.0323, Validation Accuracy: 0.9600, Train Accuracy: 0.9702, Train Time: 24.51, Eval Time: 20.18, ETA: 2024-09-15 19:50:37
Epoch 33, Train Loss: 0.0328, Validation Accuracy: 0.9663, Train Accuracy: 0.9739, Train Time: 24.57, Eval Time: 19.52, ETA: 2024-09-15 19:50:12
Epoch 34, Train Loss: 0.0307, Validation Accuracy: 0.9627, Train Accuracy: 0.9690, Train Time: 24.33, Eval Time: 19.22, ETA: 2024-09-15 19:49:50
Epoch 35, Train Loss: 0.0318, Validation Accuracy: 0.9580, Train Accuracy: 0.9669, Train Time: 24.32, Eval Time: 19.06, ETA: 2024-09-15 19:49:43
Epoch 36, Train Loss: 0.0317, Validation Accuracy: 0.9563, Train Accuracy: 0.9684, Train Time: 23.99, Eval Time: 19.25, ETA: 2024-09-15 19:49:38
Epoch 37, Train Loss: 0.0310, Validation Accuracy: 0.9653, Train Accuracy: 0.9741, Train Time: 24.61, Eval Time: 19.75, ETA: 2024-09-15 19:50:20
Epoch 38, Train Loss: 0.0305, Validation Accuracy: 0.9720, Train Accuracy: 0.9751, Train Time: 24.56, Eval Time: 19.69, ETA: 2024-09-15 19:50:16
Epoch 39, Train Loss: 0.0324, Validation Accuracy: 0.9580, Train Accuracy: 0.9697, Train Time: 25.33, Eval Time: 19.76, ETA: 2024-09-15 19:50:46
Epoch 40, Train Loss: 0.0312, Validation Accuracy: 0.9620, Train Accuracy: 0.9742, Train Time: 24.41, Eval Time: 19.57, ETA: 2024-09-15 19:50:07
Epoch 41, Train Loss: 0.0340, Validation Accuracy: 0.9683, Train Accuracy: 0.9701, Train Time: 24.39, Eval Time: 19.60, ETA: 2024-09-15 19:50:08
Epoch 42, Train Loss: 0.0340, Validation Accuracy: 0.9540, Train Accuracy: 0.9681, Train Time: 24.30, Eval Time: 19.91, ETA: 2024-09-15 19:50:15
Epoch 43, Train Loss: 0.0332, Validation Accuracy: 0.9630, Train Accuracy: 0.9728, Train Time: 24.43, Eval Time: 19.98, ETA: 2024-09-15 19:50:22
Epoch 44, Train Loss: 0.0326, Validation Accuracy: 0.9703, Train Accuracy: 0.9713, Train Time: 25.09, Eval Time: 19.75, ETA: 2024-09-15 19:50:35
Epoch 45, Train Loss: 0.0343, Validation Accuracy: 0.9640, Train Accuracy: 0.9673, Train Time: 24.91, Eval Time: 20.40, ETA: 2024-09-15 19:50:49
Epoch 46, Train Loss: 0.0317, Validation Accuracy: 0.9653, Train Accuracy: 0.9727, Train Time: 24.51, Eval Time: 19.92, ETA: 2024-09-15 19:50:23
Epoch 47, Train Loss: 0.0321, Validation Accuracy: 0.9600, Train Accuracy: 0.9732, Train Time: 24.60, Eval Time: 19.97, ETA: 2024-09-15 19:50:27
Epoch 48, Train Loss: 0.0335, Validation Accuracy: 0.9543, Train Accuracy: 0.9718, Train Time: 24.90, Eval Time: 19.68, ETA: 2024-09-15 19:50:28
Epoch 49, Train Loss: 0.0330, Validation Accuracy: 0.9667, Train Accuracy: 0.9674, Train Time: 25.09, Eval Time: 19.91, ETA: 2024-09-15 19:50:39
Epoch 50, Train Loss: 0.0322, Validation Accuracy: 0.9640, Train Accuracy: 0.9730, Train Time: 25.28, Eval Time: 19.85, ETA: 2024-09-15 19:50:42
Epoch 51, Train Loss: 0.0349, Validation Accuracy: 0.9550, Train Accuracy: 0.9658, Train Time: 25.27, Eval Time: 20.08, ETA: 2024-09-15 19:50:47
Epoch 52, Train Loss: 0.0344, Validation Accuracy: 0.9690, Train Accuracy: 0.9705, Train Time: 24.65, Eval Time: 19.77, ETA: 2024-09-15 19:50:26
Epoch 53, Train Loss: 0.0351, Validation Accuracy: 0.9550, Train Accuracy: 0.9629, Train Time: 24.61, Eval Time: 19.77, ETA: 2024-09-15 19:50:25
Epoch 54, Train Loss: 0.0358, Validation Accuracy: 0.9577, Train Accuracy: 0.9703, Train Time: 26.50, Eval Time: 20.19, ETA: 2024-09-15 19:51:13
Epoch 55, Train Loss: 0.0336, Validation Accuracy: 0.9640, Train Accuracy: 0.9707, Train Time: 24.53, Eval Time: 20.23, ETA: 2024-09-15 19:50:35
Epoch 56, Train Loss: 0.0370, Validation Accuracy: 0.9570, Train Accuracy: 0.9671, Train Time: 25.24, Eval Time: 20.37, ETA: 2024-09-15 19:50:51
Epoch 57, Train Loss: 0.0355, Validation Accuracy: 0.9597, Train Accuracy: 0.9684, Train Time: 25.16, Eval Time: 20.07, ETA: 2024-09-15 19:50:44
Epoch 58, Train Loss: 0.0319, Validation Accuracy: 0.9700, Train Accuracy: 0.9700, Train Time: 24.42, Eval Time: 19.43, ETA: 2024-09-15 19:50:21
Epoch 59, Train Loss: 0.0317, Validation Accuracy: 0.9660, Train Accuracy: 0.9716, Train Time: 24.39, Eval Time: 19.14, ETA: 2024-09-15 19:50:16
Epoch 60, Train Loss: 0.0345, Validation Accuracy: 0.9650, Train Accuracy: 0.9663, Train Time: 24.23, Eval Time: 20.44, ETA: 2024-09-15 19:50:33
Epoch 61, Train Loss: 0.0333, Validation Accuracy: 0.9673, Train Accuracy: 0.9728, Train Time: 24.33, Eval Time: 19.62, ETA: 2024-09-15 19:50:23
Epoch 62, Train Loss: 0.0348, Validation Accuracy: 0.9627, Train Accuracy: 0.9711, Train Time: 24.50, Eval Time: 20.10, ETA: 2024-09-15 19:50:31
Epoch 63, Train Loss: 0.0362, Validation Accuracy: 0.9670, Train Accuracy: 0.9700, Train Time: 24.40, Eval Time: 19.71, ETA: 2024-09-15 19:50:25
Epoch 64, Train Loss: 0.0399, Validation Accuracy: 0.9647, Train Accuracy: 0.9604, Train Time: 24.57, Eval Time: 19.42, ETA: 2024-09-15 19:50:24
Epoch 65, Train Loss: 0.0436, Validation Accuracy: 0.9600, Train Accuracy: 0.9629, Train Time: 24.32, Eval Time: 19.54, ETA: 2024-09-15 19:50:23
Epoch 66, Train Loss: 0.0392, Validation Accuracy: 0.9613, Train Accuracy: 0.9687, Train Time: 24.32, Eval Time: 19.18, ETA: 2024-09-15 19:50:19
Epoch 67, Train Loss: 0.0347, Validation Accuracy: 0.9527, Train Accuracy: 0.9619, Train Time: 24.43, Eval Time: 19.44, ETA: 2024-09-15 19:50:22
Epoch 68, Train Loss: 0.0405, Validation Accuracy: 0.9580, Train Accuracy: 0.9591, Train Time: 24.78, Eval Time: 20.07, ETA: 2024-09-15 19:50:29
Epoch 69, Train Loss: 0.0399, Validation Accuracy: 0.9557, Train Accuracy: 0.9660, Train Time: 26.21, Eval Time: 19.86, ETA: 2024-09-15 19:50:36
Epoch 70, Train Loss: 0.0406, Validation Accuracy: 0.9573, Train Accuracy: 0.9617, Train Time: 25.02, Eval Time: 20.19, ETA: 2024-09-15 19:50:32
Epoch 71, Train Loss: 0.0413, Validation Accuracy: 0.9550, Train Accuracy: 0.9623, Train Time: 24.32, Eval Time: 19.37, ETA: 2024-09-15 19:50:26
Epoch 72, Train Loss: 0.0401, Validation Accuracy: 0.9623, Train Accuracy: 0.9666, Train Time: 24.37, Eval Time: 19.52, ETA: 2024-09-15 19:50:27
Epoch 73, Train Loss: 0.0427, Validation Accuracy: 0.9683, Train Accuracy: 0.9691, Train Time: 25.83, Eval Time: 20.21, ETA: 2024-09-15 19:50:31
Epoch 74, Train Loss: 0.0379, Validation Accuracy: 0.9583, Train Accuracy: 0.9648, Train Time: 24.98, Eval Time: 19.70, ETA: 2024-09-15 19:50:30
Epoch 75, Train Loss: 0.0419, Validation Accuracy: 0.9347, Train Accuracy: 0.9308, Train Time: 24.55, Eval Time: 19.83, ETA: 2024-09-15 19:50:30
"""

model_logs = [
    ("Model big", model1_log),
    ("Model middle", model2_log),
    ("Model small", model3_log)
]
if __name__ == "__main__":
    
    compare_models(model_logs)