import os

import torch


class Config:
    # device: torch.device = torch.device("cpu")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir: str = ".cache"
    distilbert_path: str = "./distilbert-base-uncased"
    DOCKER_IMAGE_DOCUMENT_CLASSIFICATION_SERVICE = "konstantinschulz/document-classification-german:v1"
    DOCKER_PORT_CREDIBILITY = 8000
    DOCUMENT_CLASSIFICATION_SERVICE: str = "document-classification-service"
    HOST_PORT_CREDIBILITY = 8000
    label_path: str = "label_map.json"
    max_length: int = 512
    model_name: str = os.path.join(os.path.abspath("models"), "bs4_idx_5500_loss_094.pth")
    test_path: str = os.path.join(cache_dir, "test_data")
    train_path: str = os.path.join(cache_dir, "train_data")
    val_path: str = os.path.join(cache_dir, "val_data")
