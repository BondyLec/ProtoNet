import torch
import torch.nn as nn
from torch import optim

from main import class_to_indices
from utils.omniglot import OmniglotCombined
from utils.split import load_vinyals_split
from utils.episode import sample_episode, build_class_index
from models.Protonet import Protonet

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

TRAIN_WAY = 60
VAL_WAY = 5
K_SHOT = 1

TRAIN_QUERY = 5
VAL_QUERY = 5

NUM_EPISODES = 10000

VAL_INTERVAL = 100
VAL_EPISODES = 100

LEARNING_RATE = 1e-3

dataset = OmniglotCombined(
    root="./data"
)
class_to_indices = build_class_index(dataset)
meta_train_class = load_vinyals_split(
    "train",
    dataset.name_to_label
)
meta_val_class = load_vinyals_split(
    "val",
    dataset.name_to_label
)
print(f"Meta training classes: {len(meta_train_class)}")
print(f"Meta validation classes: {len(meta_val_class)}")

model = Protonet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 每2000个episode将学习率减半
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

def evaluate_validation():
    model.eval()
    total_correct = 0
    total_query = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(NUM_EPISODES):
            support_x, support_y, query_x, query_y, _ = sample_episode(
                dataset,
                class_to_indices,
                meta_val_class,
                n_way=VAL_WAY,
                k_shot=K_SHOT,
                q_query=VAL_QUERY
            )

            support_x = support_x.to(device)
            support_y = support_y.to(device)

            query_x = query_x.to(device)
            query_y = query_y.to(device)

            logits = model(
                support_x,
                support_y,
                query_x,
                n_way=VAL_WAY
            )

            loss = criterion(logits, query_y)
            pred = logits.argmax(dim=1)
            total_correct += (pred == query_y).sum().item()
            total_query += query_y.size(0)
            total_loss += loss.item()
    avg_accuracy = total_correct / total_query
    avg_loss = total_loss / NUM_EPISODES
    return avg_accuracy, avg_loss

best_val_accuracy = 0.0

train_loss_history = []
train_acc_history = []

for episode in range(NUM_EPISODES):
    model.train()

    support_x, support_y, query_x, query_y, _ = sample_episode(
        dataset,
        class_to_indices,
        meta_train_class,
        n_way=TRAIN_WAY,
        k_shot=K_SHOT,
        q_query=TRAIN_QUERY
    )

    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)
    # 反向传播
    optimizer.zero_grad()
    logits = model(
        support_x,
        support_y,
        query_x,
        n_way=TRAIN_WAY
    )
    loss = criterion(logits, query_y)
    # 计算准确率
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        accuracy = (pred == query_y).float().mean().item()

    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss_history.append(loss.item())
    train_acc_history.append(accuracy)

    if episode % VAL_INTERVAL == 0:
        avg_train_loss = sum(train_loss_history[-VAL_INTERVAL:]) / VAL_INTERVAL
        avg_train_acc = sum(train_acc_history[-VAL_INTERVAL:]) / VAL_INTERVAL
        val_accuracy, val_loss = evaluate_validation()
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Episode {episode} | "            
            f"LR: {current_lr:.6f}\n"
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc:.4f}\n"
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
        )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "./checkpoints/best_1shot.pt")
        print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")