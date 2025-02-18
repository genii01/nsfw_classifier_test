import os
import json
import pandas as pd
from omegaconf import OmegaConf


def load_config(config_path):
    return OmegaConf.load(config_path)


def create_chess_dataframe(data_dir):
    image_paths = []
    labels = []

    # Walk through all directories
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_name)

    # Create DataFrame
    df = pd.DataFrame({"image_path": image_paths, "label": labels})

    # Convert labels to numerical values
    df["label_id"] = pd.Categorical(df["label"]).codes

    # Create label mapping dictionary
    label_mapping = {
        "labels": {
            label: int(label_id)
            for label, label_id in zip(
                df["label"].unique(), pd.Categorical(df["label"].unique()).codes
            )
        },
        "id2label": {
            str(label_id): label
            for label, label_id in zip(
                df["label"].unique(), pd.Categorical(df["label"].unique()).codes
            )
        },
    }

    return df, label_mapping


def main():
    config = load_config("config/train_config.yaml")

    data_dir = config.data.train_dir
    save_path = config.data.csv_path

    # Get directory path from csv_path
    base_dir = os.path.dirname(save_path)

    # Create label mapping json path
    label_mapping_path = os.path.join(base_dir, "label_mapping.json")

    # Create DataFrame and label mapping
    df, label_mapping = create_chess_dataframe(data_dir)

    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Save DataFrame to CSV
    df.to_csv(save_path, index=False)

    # Save label mapping to JSON
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    print(f"DataFrame saved to {save_path}")
    print(f"Label mapping saved to {label_mapping_path}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nLabel mapping:")
    print(json.dumps(label_mapping, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
