# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard --logdir logs
import gc
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DistilBertModel
from transformers.modeling_outputs import BaseModelOutput
from sklearn.metrics import classification_report, f1_score
from config import Config
from get_data import BlurbDataset, init_cache

batch_size: int = 2  # 16
gc.collect()
torch.cuda.empty_cache()
loss_bce: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')  # DistilBertForSequenceClassification
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, 8)
        self.leakyRelu: torch.nn.LeakyReLU = torch.nn.LeakyReLU()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        output_1: BaseModelOutput = self.l1(input_ids, attention_mask=attention_mask)
        output_2: torch.Tensor = self.l2(output_1.last_hidden_state)
        activation_1: torch.Tensor = self.leakyRelu(output_2)
        output: torch.Tensor = self.l3(activation_1)
        outputs_averaged: torch.Tensor = torch.mean(output, 1)  # (batch_size, num_labels)
        normalized: torch.Tensor = torch.sigmoid(outputs_averaged)
        return normalized


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor):
    return loss_bce(outputs, targets)


LEARNING_RATE = 5e-5
EPOCHS = 1
LOSS_STEP_TRAIN = 1
LOSS_STEP_VAL = 5
DATASET_SIZE_VAL = 3
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter()


def train(epoch: int, model: BERTClass, optimizer):
    train_dataset: BlurbDataset = BlurbDataset(Config.train_path)
    val_dataset: BlurbDataset = BlurbDataset(Config.val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 16
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    running_loss_train: float = 0
    for idx_train, batch in enumerate(train_loader, 0):
        outputs = model(**batch)  # (batch_size, dim, num_labels)
        optimizer.zero_grad()
        loss = loss_fn(outputs, batch["labels"])
        running_loss_train += loss.item()
        if idx_train % LOSS_STEP_TRAIN == 0:
            # ...log the running loss
            writer.add_scalar('train loss', running_loss_train / LOSS_STEP_TRAIN, epoch * len(train_loader) + idx_train)
            running_loss_train = 0
            if idx_train % LOSS_STEP_VAL == 0:
                running_loss_val: float = 0
                for idx_val, val_batch in enumerate(val_loader, 0):
                    val_outputs = model(**val_batch)
                    val_loss = loss_fn(val_outputs, val_batch["labels"])
                    running_loss_val += val_loss.item()
                    if idx_val == DATASET_SIZE_VAL:
                        break
                writer.add_scalar('val loss', running_loss_val / DATASET_SIZE_VAL,
                                  epoch * len(train_loader) + idx_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(epoch: int, model: BERTClass):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    test_dataset: BlurbDataset = BlurbDataset(Config.test_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    model.eval()
    running_loss_test: float = 0
    dataset_size: int = 0
    y_true = []
    y_pred = []
    confidences: List[float] = []
    label_map: Dict[str, str] = json.load(open(Config.label_path))
    y_pred_dict: Dict[str, List[float]] = {value: [] for value in label_map.values()}
    for idx_test, batch in tqdm(enumerate(test_loader, 0)):
        if len(y_pred) > 300:
            break
        dataset_size = idx_test + 1
        outputs = model(**batch)  # (batch_size, dim, num_labels)
        labels_target = batch["labels"]
        loss = loss_fn(outputs, labels_target)
        running_loss_test += loss.item()
        for i in range(batch_size):
            if len(labels_target) > i:
                y_true.append(torch.argmax(labels_target[i]).item())
                pred_idx: int = torch.argmax(outputs[i]).item()
                label: str = label_map[str(pred_idx)]
                y_pred.append(pred_idx)
                confidence: float = outputs[i][pred_idx].item()
                y_pred_dict[label].append(confidence)
                confidences.append(confidence)
                # if y_true[-1] != y_pred[-1]:
                #     print(outputs[i], labels_target[i])
        if idx_test % 40 == 0:
            writer.add_scalar('test loss', running_loss_test / dataset_size, epoch * len(test_loader) + dataset_size)
    print(running_loss_test / dataset_size)
    print(classification_report(y_true, y_pred))
    print(f1_score(y_true, y_pred, average="micro"))
    # Multiple box plots on one Axes
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    label_indices_ordered: List[int] = list(range(8))
    labels_ordered: List[str] = [label_map[str(i)] for i in label_indices_ordered]
    ax.set_xticklabels(labels_ordered, rotation=70)
    ax.boxplot([y_pred_dict[lo] for lo in labels_ordered])
    ax.set_title("Document Classification")
    ax.set_xlabel("Predicted Document Type")
    ax.set_ylabel("Confidence of AI Decision")
    fig.tight_layout(pad=2)
    plt.show()


if __name__ == "__main__":
    model = BERTClass()
    model.load_state_dict(torch.load(Config.model_name))
    model.to(Config.device)
    if not all(os.path.exists(x) for x in [Config.train_path, Config.val_path, Config.test_path]):
        init_cache()
    for epoch in range(EPOCHS):
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        # train(epoch, model, optimizer)
        test(epoch, model)
# torch.save(model.state_dict(), Config.model_name)
