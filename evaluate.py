import math
import torch

from models.Protonet import Protonet
from utils.omniglot import OmniglotCombined
from utils.split import load_vinyals_split
from utils.episode import sample_episode, build_class_index

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.cuda.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

N_WAY = 5
K_SHOT = 1
Q_QUERY = 15
NUM_TEST_EPISODES = 1000
dataset = OmniglotCombined(
    root="./data"
)
class_to_indices = build_class_index(dataset)
meta_test_classes = load_vinyals_split(
    "test",
    dataset.name_to_label
)
print(f"Meta testing classes: {len(meta_test_classes)}")

model = Protonet().to(device)
checkpoint = torch.load(
    "./checkpoints/best_1shot.pt",
    map_location=device
)
model.load_state_dict(checkpoint)
model.eval()

episode_accuracies = []
with torch.no_grad():
    for episode in range(NUM_TEST_EPISODES):
        support_x, support_y, query_x, query_y, selected_classes = (
            sample_episode(
                dataset=dataset,
                class_to_indices=class_to_indices,
                available_classes=meta_test_classes,
                n_way=N_WAY,
                k_shot=K_SHOT,
                q_query=Q_QUERY
            )
        )
        support_x = support_x.to(device)
        support_y = support_y.to(device)

        query_x = query_x.to(device)
        query_y = query_y.to(device)

        logits = model(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            n_way=N_WAY
        )
        pred = logits.argmax(1)
        accuracy = (pred == query_y).float().mean().item()
        episode_accuracies.append(accuracy)

mean_accuracy = (sum(episode_accuracies) / len(episode_accuracies))
variance = sum(
    (acc - mean_accuracy) ** 2 for acc in episode_accuracies
) / (len(episode_accuracies) - 1)
std = math.sqrt(variance)

confidence_interval = 1.96 * std / math.sqrt(len(episode_accuracies))
print(
    "Meta test results:\n"
    f"Mean accuracy: {mean_accuracy * 100:.2f}\n"
    f"Std: {std * 100:.2f}\n"
    f"95% confidence interval: ±{confidence_interval * 100:.2f}"
    f"Final: {mean_accuracy * 100:.2f} ± {confidence_interval * 100:.2f}"
)