from config import cfg
from dataset import load_dataframe, CustomDataset
from experiment import run_experiment
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow
import timm


# Load and split data
df = load_dataframe(cfg["root_dir"])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=cfg["seed"])

# Define transforms
basic_transform = A.Compose([
    A.Resize(cfg["image_size"], cfg["image_size"]),
    ToTensorV2()
])

# Build datasets
train_dataset = CustomDataset(cfg, train_df, transform=basic_transform)
valid_dataset = CustomDataset(cfg, val_df, transform=basic_transform)

# Define search space
backbones = ['resnet18', 'efficientnet_b0', 'mobilenetv3_small']
batch_sizes = [16, 32]
learning_rates = [1e-3, 5e-4, 1e-4]

def config_already_ran(exp_id, backbone, batch_size, learning_rate):
    runs = mlflow.search_runs(experiment_ids=[exp_id])
    
    return any(
        (runs["params.backbone"] == backbone) &
        (runs["params.batch_size"] == str(batch_size)) &  # note: these are stored as strings in MLflow
        (runs["params.learning_rate"] == str(learning_rate))
    )


mlflow.set_experiment("FER_Tuning_Compact_Models")
exp = mlflow.get_experiment_by_name("FER_Tuning_Compact_Models")

batch_sizes = [16, 32]
learning_rates = [1e-3, 5e-4, 1e-4]


# Your desired models
candidate_backbones = ['resnet18', 'efficientnet_b0', 'mobilenetv3_small', 'mobilenetv3_rw', 'mobilenetv3_large_100']

# Get all available models from timm
available_models = timm.list_models()

# Filter the ones that actually exist
backbones = [m for m in candidate_backbones if m in available_models]

if not backbones:
    raise ValueError("No valid backbones found in your timm installation.")
else:
    print(f"Valid backbones for this run: {backbones}")


for backbone in backbones:
    for bs in batch_sizes:
        for lr in learning_rates:
            if config_already_ran(exp.experiment_id, backbone, bs, lr):
                print(f"Skipping already-run config: {backbone}, BS={bs}, LR={lr}")
                continue
            run_experiment(cfg, train_dataset, valid_dataset, backbone, bs, lr)
