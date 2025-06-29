import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, criterion, device, dataset_name):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            batch_y = batch_y.long()

            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"{dataset_name}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    # 保存分类报告
    report = classification_report(all_targets, all_predicted, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'reports/{dataset_name}_classification_report.csv')

    return test_loss / len(test_loader), accuracy


def plot_results(train_losses, train_accuracies, test_losses, test_accuracies, num_epochs, dataset_name):
    plt.figure(figsize=(12, 4))

    # 绘制训练和测试的损失
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()

    # 绘制训练和测试的准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name}_training_plot.png')




