from typing import Any, Dict

from annotate import get_topic
from config import Config
from elg import FlaskService
from elg.model import ClassificationResponse


class DocumentClassificationService(FlaskService):

    def convert_outputs(self, content: str) -> ClassificationResponse:
        topic_dict: Dict[str, float] = get_topic(content)
        return ClassificationResponse(classes=[{"class": k, "score": v} for k, v in topic_dict.items()])

    def process_text(self, content: Any) -> ClassificationResponse:
        return self.convert_outputs(content.content)


dcs: DocumentClassificationService = DocumentClassificationService(Config.DOCUMENT_CLASSIFICATION_SERVICE)
app = dcs.app
