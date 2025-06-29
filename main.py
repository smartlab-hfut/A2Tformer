import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from dataprosess import prepare_data, get_loaders
from model import Transformer  # 保证和你的主模型一致
from test_model import evaluate_model, plot_results
import pandas as pd


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


def train_and_evaluate(dataset_name, model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs,
                       num_classes, device):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_test_accuracy, best_epoch = 0.0, 0
    result_file_name = "best_test_accuracies_binary.txt" if num_classes == 2 else "best_test_accuracies_multi.txt"
    result_file = open(result_file_name, "a")

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        # --- Save features for visualization ---
        if epoch == num_epochs - 1:
            model.eval()
            all_feats_before, all_feats_after, all_labels = [], [], []

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    block = model.blocks[0]
                    x = block.attn(batch_x)
                    x_before, x_after = block.ffn(x, return_features=True)

                    all_feats_before.append(x_before.cpu().numpy())
                    all_feats_after.append(x_after.cpu().numpy())
                    all_labels.append(batch_y.cpu().numpy())

            # 合并所有 batch
            feats_before = np.concatenate(all_feats_before, axis=0)
            feats_after = np.concatenate(all_feats_after, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            # 平均池化（如有需要）
            # feats_before = feats_before.mean(axis=1)
            # feats_after = feats_after.mean(axis=1)

            np.save(f"./viewdata/{dataset_name}_features_before.npy", feats_before)
            np.save(f"./viewdata/{dataset_name}_features_after.npy", feats_after)
            np.save(f"./viewdata/{dataset_name}_labels.npy", labels)

            print(f"[✓] Features saved for {dataset_name}: shape={feats_before.shape}")

        # --- Test ---
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(100 * correct / total)

        if test_accuracies[-1] > best_test_accuracy:
            best_test_accuracy = test_accuracies[-1]
            best_epoch = epoch + 1

        print(
            f"{dataset_name}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.2f}%")

    result_file.write(f"{dataset_name}: Best Test Accuracy: {best_test_accuracy:.2f}% (Epoch {best_epoch})\n")
    result_file.close()
    return train_losses, train_accuracies, test_losses, test_accuracies

all_results = {}  # 新增：记录每个数据集的结果
# --- Main ---
if __name__ == '__main__':
    try:
        with open('config.json', 'r') as f:
            datasets_info = json.load(f)
        for dataset_name, config in datasets_info.items():
            try:
                print(
                    f"\n=== Training {dataset_name} (seq_len={config['time_length']}, classes={config['num_classes']}) ===")
                # 数据
                x_train, y_train, x_test, y_test = prepare_data(dataset_name)
                train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test,
                                                        batch_size=config["batch_size"])
                # 模型
                model = Transformer(
                    dimension=config["dimension"], d_hid=config["d_hid"], d_inner=1024,
                    n_layers=config["n_layer"], num_layers=1, dropout=config.get("dropout", 0),
                    class_num=config["num_classes"]
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
                num_epochs = config["epochs"]
                # 训练
                train_losses, train_accuracies, test_losses, test_accuracies = train_and_evaluate(
                    dataset_name, model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs,
                    config["num_classes"], device)
                best_acc = max(test_accuracies)
                all_results[dataset_name] = {'best_accuracy': best_acc}
                # 最终测试
                test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, dataset_name)
                # 可选可视化
                # plot_results(train_losses, train_accuracies, test_losses, test_accuracies, num_epochs, dataset_name)
            except Exception as e:
                print(f"[Warning] {dataset_name} 跑不通，跳过。Error: {e}")
                import traceback

                traceback.print_exc()
                continue  # 关键：跳到下一个数据集

        if all_results:
            df = pd.DataFrame.from_dict(all_results, orient='index')
            df.index.name = 'dataset'
            df.to_excel('all_results.xlsx')
            print('所有数据集最优accuracy已保存为 all_results.xlsx')


    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
