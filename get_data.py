import os
from bs4 import BeautifulSoup
from bs4.element import PageElement, ResultSet

def read_data_from_file(file_path: str):
    texts = []
    labels = []
    file_content: str = open(file_path, encoding="utf-8").read()
    soup: BeautifulSoup = BeautifulSoup(file_content, "lxml")
    books: ResultSet = soup.find_all("book")
    for book in books:
        element: PageElement = book
        texts.append(element.contents[2][1:-1])
        category: PageElement = element.find_next("category")
        top_level_topics: ResultSet = category.select('topic[d="0"]')
        labels.append(top_level_topics[0].get_text())
    return texts, labels

data_dir: str = os.path.abspath("data")
train_file: str = os.path.join(data_dir, "blurbs_train.txt")
dev_file: str = os.path.join(data_dir, "blurbs_dev.txt")
train_texts, train_labels = read_data_from_file(train_file)
