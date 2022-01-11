import json
from typing import Dict, List

import numpy
import torch
from transformers import DistilBertTokenizerFast

from config import Config
from get_data import read_data_from_file, transform_labels
from train import BERTClass

model = BERTClass()
model.load_state_dict(torch.load(Config.model_name))
model.to(Config.device)
model.eval()
tokenizer: DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(Config.distilbert_path)


def get_label_map() -> None:
    texts, labels_verbal = read_data_from_file("data/blurbs_dev.txt")
    labels_numpy: List[numpy.ndarray] = transform_labels([labels_verbal])[0]
    labels_numeric: List[str] = [x.argmax().item() for x in labels_numpy]
    label_dict: Dict[str, str] = dict()
    for i in range(len(labels_numeric)):
        if len(label_dict) == 8:
            break
        if labels_numeric[i] not in label_dict:
            label_dict[labels_numeric[i]] = labels_verbal[i]
    json.dump(label_dict, open(Config.label_path, "w+"))


def get_topic(text: str) -> Dict[str, float]:
    label_map: Dict[str, str]
    with open(Config.label_path) as f:
        label_map = json.load(f)
    encoding = tokenizer([text], truncation=True, padding=True, max_length=Config.max_length)
    encoding_tensors: Dict[str, torch.Tensor] = {k: torch.tensor(v, device=Config.device) for k, v in encoding.items()}
    output: torch.Tensor = model(**encoding_tensors)  # (batch_size, num_labels)
    y_probs: torch.Tensor = output.squeeze()  # (num_labels)
    pred_dict: Dict[str, float] = {label_map[str(i)]: y_probs[i].item() for i in range(len(y_probs))}
    return pred_dict
