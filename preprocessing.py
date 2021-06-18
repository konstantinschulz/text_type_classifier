import os
from lxml import etree

data_dir: str = os.path.abspath("GermEval2019T1_public_data_final")
train_file: str = os.path.join(data_dir, "blurbs_train.txt")
xml = etree.parse(train_file)
a = 0
