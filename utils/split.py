from pathlib import Path


SPLIT_DIR = Path("data/splits/vinyals")


def load_vinyals_split(split_name, name_to_label):

    split_file = SPLIT_DIR / f"{split_name}.txt"

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    virtual_classes = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # 例如：
        # Alphabet/character01/rot090

        base_name, rotation = line.rsplit(
            "/rot",
            1
        )

        angle = int(rotation)

        base_label = name_to_label[
            base_name
        ]

        virtual_classes.append(
            (base_label, angle)
        )

    return virtual_classes