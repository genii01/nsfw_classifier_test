import os
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
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_name)

    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })

    # Convert labels to numerical values
    df['label_id'] = pd.Categorical(df['label']).codes

    return df


def main():
    config = load_config('config/train_config.yaml')

    data_dir = config.data.train_dir
    save_path = config.data.csv_path

    # Create DataFrame
    df = create_chess_dataframe(data_dir)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"DataFrame saved to {save_path}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df['label'].value_counts())


if __name__ == "__main__":
    main()
