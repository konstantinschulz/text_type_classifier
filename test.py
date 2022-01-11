from typing import Any

from config import Config
import time
import docker
from docker.models.containers import Container
from elg.model.response.ClassificationResponse import ClassificationResponse, ClassesResponse
from elg.service import Service
from elg.model import TextRequest
import unittest

from elg_service import DocumentClassificationService


class ElgTestCase(unittest.TestCase):
    class_field: str = "Glaube & Ethik"
    content: str = "Die Konfirmandenzeit wird für Jugendliche besonders dann zu einer nachhaltigen Erfah­rung, wenn ihre Eltern Anteil nehmen und sie hilfreich begleiten. Um sie dabei zu unterstützen, bietet diese Broschüre praktische Hinweise zu Formen und Organisation der Konfirmandenzeit in den Gemeinden, Bilder und Berichte zur religiösen und pädagogischen Gestaltung der Konfirmandenarbeit heute sowie Hinweise zur Vorbereitung und Feier der Konfirmation in der Familie. Besonders für die Eltern, für die die Konfirmation ihrer Kinder eine Wiederbegegnung mit Kirche ist, hält dieses Heft zudem Erklärungen zu den Festen und Feiertagen der Kirche, Antworten auf häufig gestellte Fragen und ein kleines Glossar kirchlicher Begriffe bereit. Mit diesem Heft wird die Konfirmandenzeit zu einer Bereicherung auch für die Eltern."
    score: float = 0.6543954014778137

    def test_local(self):
        request = TextRequest(content=ElgTestCase.content)
        service = DocumentClassificationService(Config.DOCUMENT_CLASSIFICATION_SERVICE)
        response = service.process_text(request)
        max_cr: ClassesResponse = max(response.classes, key=lambda x: x.score)
        self.assertEqual(max_cr.class_field, ElgTestCase.class_field)
        self.assertEqual(max_cr.score, ElgTestCase.score)
        self.assertEqual(type(response), ClassificationResponse)

    def test_docker(self):
        client = docker.from_env()
        ports_dict: dict = dict()
        ports_dict[Config.DOCKER_PORT_CREDIBILITY] = Config.HOST_PORT_CREDIBILITY
        container: Container = client.containers.run(
            Config.DOCKER_IMAGE_DOCUMENT_CLASSIFICATION_SERVICE, ports=ports_dict, detach=True)
        # wait for the container to start the API
        time.sleep(1)
        service: Service = Service.from_docker_image(
            Config.DOCKER_IMAGE_DOCUMENT_CLASSIFICATION_SERVICE,
            f"http://localhost:{Config.DOCKER_PORT_CREDIBILITY}/process", Config.HOST_PORT_CREDIBILITY)
        response: Any = service(ElgTestCase.content, sync_mode=True)
        container.stop()
        container.remove()
        max_cr: ClassesResponse = max(response.classes, key=lambda x: x.score)
        self.assertEqual(max_cr.class_field, ElgTestCase.class_field)
        self.assertEqual(max_cr.score, ElgTestCase.score)
        self.assertEqual(type(response), ClassificationResponse)

    def test_elg_remote(self):
        service = Service.from_id(7484)
        response: Any = service(ElgTestCase.content)
        max_cr: ClassesResponse = max(response.classes, key=lambda x: x.score)
        self.assertEqual(max_cr.class_field, ElgTestCase.class_field)
        self.assertEqual(max_cr.score, ElgTestCase.score)
        self.assertEqual(type(response), ClassificationResponse)


if __name__ == '__main__':
    unittest.main()
