from clams.serve import ClamsApp
from clams.restify import Restifier

from east_utils import *

APP_VERSION = 0.1


class EAST_td(ClamsApp):
    def setupmetadata(self):
        return {"name": "EAST Text Detection",
                "description": "This tool applies EAST test detection to the video or image.",
                "vendor": "Team CLAMS",
                "iri": f"http://mmif.clams.ai/apps/east/{APP_VERSION}",
                "requires": [DocumentTypes.ImageDocument, DocumentTypes.VideoDocument],
                "produces": [AnnotationTypes.BoundingBox]}

    def sniff(self, mmif):
        # this mock-up method always returns true
        return True

    def annotate(self, mmif: Mmif) -> str:
        new_view = mmif.new_view()
        new_view.metadata['app'] = self.metadata["iri"]

        if mmif.get_documents_by_type(DocumentTypes.VideoDocument.value):
            mmif = run_EAST_video(mmif, new_view)
        elif mmif.get_documents_by_type(DocumentTypes.ImageDocument.value):
            mmif = run_EAST_image(mmif, new_view)
        return str(mmif)


if __name__ == "__main__":
    td_tool = EAST_td()
    td_service = Restifier(td_tool)
    td_service.run()
