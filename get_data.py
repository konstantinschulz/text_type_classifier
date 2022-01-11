import os
import pickle
from typing import List, Dict, Tuple

import numpy
from bs4 import BeautifulSoup
from bs4.element import PageElement, ResultSet
from torch.utils.data import Subset
from transformers import DistilBertTokenizerFast
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from config import Config


class BlurbDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path: str, encodings: dict = None, labels: list = None):
        self.folder_path: str = folder_path
        self.attention_mask_file_name: str = "attention_mask.pickle"
        self.input_ids_file_name: str = "input_ids.pickle"
        self.label_file_name: str = "label.pickle"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            input_ids: list = encodings["input_ids"]
            attention_mask: list = encodings["attention_mask"]
            for i in range(len(labels)):
                subfolder: str = os.path.join(self.folder_path, str(i))
                os.makedirs(subfolder)
                pickle.dump(attention_mask[i], open(os.path.join(subfolder, self.attention_mask_file_name), "wb+"))
                pickle.dump(input_ids[i], open(os.path.join(subfolder, self.input_ids_file_name), "wb+"))
                pickle.dump(labels[i], open(os.path.join(subfolder, self.label_file_name), "wb+"))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        subfolder: str = os.path.join(self.folder_path, str(idx))
        attention_mask = pickle.load(open(os.path.join(subfolder, self.attention_mask_file_name), "rb"))
        input_ids = pickle.load(open(os.path.join(subfolder, self.input_ids_file_name), "rb"))
        labels = pickle.load(open(os.path.join(subfolder, self.label_file_name), "rb"))
        item: Dict[str, torch.Tensor] = {
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int8).to(Config.device),
            "input_ids": torch.tensor(input_ids, dtype=torch.int64).to(Config.device),
            "labels": torch.tensor(labels, dtype=torch.float16).to(Config.device)}
        return item

    def __len__(self):
        return len([0 for x in os.listdir(self.folder_path)])


def read_data_from_file(file_path: str) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels_verbal: List[str] = []
    file_content: str = open(file_path, encoding="utf-8").read()
    soup: BeautifulSoup = BeautifulSoup(file_content, "lxml")
    books: ResultSet = soup.find_all("book")
    for book in books:
        element: PageElement = book
        texts.append(element.contents[2][1:-1])
        category: PageElement = element.find_next("category")
        top_level_topics: ResultSet = category.select('topic[d="0"]')
        labels_verbal.append(top_level_topics[0].get_text())
    return texts, labels_verbal


def transform_labels(all_labels: List[List[str]]) -> List[List[numpy.ndarray]]:
    labels_verbal: List[str] = [y for x in all_labels for y in x]
    mlb: MultiLabelBinarizer = MultiLabelBinarizer()
    labels_numeric: List[numpy.ndarray] = mlb.fit_transform([{x} for x in labels_verbal])
    ret_val: List[List[numpy.ndarray]] = []
    for i in range(len(all_labels)):
        start_idx: int = sum(len(x) for x in ret_val)
        end_idx: int = start_idx + len(all_labels[i])
        ret_val.append(labels_numeric[start_idx:end_idx])
    return ret_val


def init_cache() -> None:
    os.makedirs(Config.cache_dir, exist_ok=True)
    data_dir: str = os.path.abspath("data")
    train_file: str = os.path.join(data_dir, "blurbs_train.txt")
    dev_file: str = os.path.join(data_dir, "blurbs_dev.txt")
    test_file: str = os.path.join(data_dir, "blurbs_test.txt")
    train_texts, train_labels = read_data_from_file(train_file)
    val_texts, val_labels = read_data_from_file(dev_file)
    test_texts, test_labels = read_data_from_file(test_file)
    train_labels, val_labels, test_labels = transform_labels([train_labels, val_labels, test_labels])

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=Config.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=Config.max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=Config.max_length)

    train_dataset: BlurbDataset = BlurbDataset(Config.train_path, train_encodings, train_labels)
    val_dataset: BlurbDataset = BlurbDataset(Config.val_path, val_encodings, val_labels)
    test_dataset: BlurbDataset = BlurbDataset(Config.test_path, test_encodings, test_labels)
