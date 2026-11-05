from utils.omniglot import OmniglotCombined
from utils.split import load_vinyals_split
from utils.episode import sample_episode, build_class_index
from models.Protonet import Protonet

dataset = OmniglotCombined(root="./data")
class_to_indices = build_class_index(dataset)

meta_train_classes = load_vinyals_split("train", dataset.name_to_label)
meta_val_classes = load_vinyals_split("val", dataset.name_to_label)
meta_trainval_classes = load_vinyals_split("trainval", dataset.name_to_label)
meta_test_classes = load_vinyals_split("test", dataset.name_to_label)
print(f"Meta training classes: {len(meta_train_classes)}")
print(f"Meta validation classes: {len(meta_val_classes)}")
print(f"Meta trainval classes: {len(meta_trainval_classes)}")
print(f"Meta testing classes: {len(meta_test_classes)}")

support_x, support_y, query_x, query_y, selected_classes = (
    sample_episode(
        dataset=dataset,
        class_to_indices=class_to_indices,
        available_classes=meta_train_classes,
        n_way=5,
        k_shot=1,
        q_query=5
    )
)

model = Protonet()
logits = model(
    support_x=support_x,
    support_y=support_y,
    query_x=query_x,
    n_way=5
)
print(f"Logits shape:{logits.shape}")
pred = logits.argmax(1)
print(f"Predictions: {pred}")
print(f"Real: {query_y}")
accuracy = (pred == query_y).float().mean().item()
print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    print("done")
