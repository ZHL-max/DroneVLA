"""
结果绘图脚本

生成训练和评估的可视化图表

使用方法：
    python scripts/plot_results.py --experiment iteration_01
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse


def plot_training_summary(exp_dir):
    """绘制训练总结"""
    metrics_path = f"{exp_dir}/logs/metrics.json"
    if not os.path.exists(metrics_path):
        print(f"未找到 {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：损失对比
    labels = ['Train Loss', 'Val Loss']
    values = [metrics.get('final_train_loss', 0), metrics.get('best_val_loss', 0)]
    colors = ['#2196F3', '#FF5722']
    bars = axes[0].bar(labels, values, color=colors, width=0.5)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Results')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12)

    # 右图：训练信息
    info_text = f"""
Training Configuration:
  Epochs: {metrics.get('epochs', 'N/A')}
  Device: {metrics.get('device', 'N/A')}
  Parameters: {metrics.get('total_params', 'N/A'):,}
  Training Time: {metrics.get('training_time', 0):.0f}s

Best Results:
  Best Val Loss: {metrics.get('best_val_loss', 'N/A'):.4f}
  Final Train Loss: {metrics.get('final_train_loss', 'N/A'):.4f}
"""
    axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Experiment Info')

    plt.tight_layout()
    os.makedirs(f"{exp_dir}/visualizations", exist_ok=True)
    plt.savefig(f"{exp_dir}/visualizations/training_summary.png", dpi=150)
    plt.close()
    print(f"训练总结已保存: {exp_dir}/visualizations/training_summary.png")


def plot_evaluation(exp_dir):
    """绘制评估结果"""
    eval_path = f"{exp_dir}/logs/eval_results.json"
    if not os.path.exists(eval_path):
        print(f"未找到 {eval_path}")
        return

    with open(eval_path) as f:
        results = json.load(f)

    tasks = list(results.keys())
    success_rates = [results[t]['success_rate'] for t in tasks]
    avg_steps = [results[t]['avg_steps'] for t in tasks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 成功率
    colors = ['#4CAF50' if r > 50 else '#FF9800' if r > 0 else '#F44336' for r in success_rates]
    bars = axes[0].bar(tasks, success_rates, color=colors, width=0.5)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Task Success Rate')
    axes[0].set_ylim(0, 110)
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12)

    # 平均步数
    bars = axes[1].bar(tasks, avg_steps, color='#2196F3', width=0.5)
    axes[1].set_ylabel('Average Steps')
    axes[1].set_title('Average Steps to Complete')
    for bar, val in zip(bars, avg_steps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    os.makedirs(f"{exp_dir}/visualizations", exist_ok=True)
    plt.savefig(f"{exp_dir}/visualizations/evaluation_results.png", dpi=150)
    plt.close()
    print(f"评估结果已保存: {exp_dir}/visualizations/evaluation_results.png")


def plot_comparison():
    """对比所有实验"""
    experiments = sorted([d for d in os.listdir("experiments") if d.startswith("iteration_")])
    if len(experiments) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 收集数据
    val_losses = []
    exp_names = []
    for exp in experiments:
        metrics_path = f"experiments/{exp}/logs/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            val_losses.append(metrics.get('best_val_loss', 0))
            exp_names.append(exp)

    if val_losses:
        axes[0].bar(exp_names, val_losses, color='#FF5722', width=0.5)
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Best Validation Loss Comparison')
        for i, val in enumerate(val_losses):
            axes[0].text(i, val, f'{val:.4f}', ha='center', va='bottom')

    # 收集成功率
    all_tasks = set()
    all_results = {}
    for exp in experiments:
        eval_path = f"experiments/{exp}/logs/eval_results.json"
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                results = json.load(f)
            all_results[exp] = results
            all_tasks.update(results.keys())

    if all_results:
        tasks = sorted(all_tasks)
        x = range(len(tasks))
        width = 0.8 / len(all_results)
        for i, (exp, results) in enumerate(all_results.items()):
            values = [results.get(t, {}).get('success_rate', 0) for t in tasks]
            axes[1].bar([xi + i * width for xi in x], values, width, label=exp)

        axes[1].set_xlabel('Task')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].set_title('Success Rate Comparison')
        axes[1].set_xticks([xi + width * len(all_results) / 2 for xi in x])
        axes[1].set_xticklabels(tasks)
        axes[1].legend()
        axes[1].set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig("experiments/comparison.png", dpi=150)
    plt.close()
    print("实验对比已保存: experiments/comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="iteration_01")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print(f"生成 {args.experiment} 的可视化...")
    plot_training_summary(f"experiments/{args.experiment}")
    plot_evaluation(f"experiments/{args.experiment}")

    if args.all:
        plot_comparison()

    print("\n可视化完成!")


if __name__ == "__main__":
    main()
