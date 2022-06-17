from clams.app import ClamsApp, AppMetadata
from clams.restify import Restifier

from east_utils import *

APP_VERSION = 0.1


class EAST_td(ClamsApp):

    def _appmetadata(self):
        metadata = {
            "name": "EAST Text Detection",
            "description": "This tool applies EAST test detection to the video or image.",
            "app_version": str(APP_VERSION),
            "app_license": "MIT",
            "url": f"http://mmif.clams.ai/apps/east/{APP_VERSION}",
            "identifier": f"http://mmif.clams.ai/apps/east/{APP_VERSION}",
            "input": [{"@type": DocumentTypes.VideoDocument, "required": True}],
            "output": [{"@type": AnnotationTypes.BoundingBox, "properties": {"boxType": "string"}},
                       {"@type": AnnotationTypes.Alignment, "properties":{}},
                       {"@type": AnnotationTypes.TimePoint, "properties":{}}
            ],
            "parameters": [
                {
                    "name": "timeUnit",
                    "type": "string",
                    "choices": ["frames", "milliseconds"],
                    "default": "frames",
                    "description": "Unit for output timepoint.",
                },
                {
                    "name": "sampleRatio",
                    "type": "integer",
                    "default": "30",
                    "description": "Frequency to sample frames.",
                },
            ],
        }
        return AppMetadata(**metadata)

    def _annotate(self, mmif: Mmif, **kwargs) -> str:
        new_view = mmif.new_view()
        config = self.get_configuration(**kwargs)
        self.sign_view(new_view, config)
        if mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            mmif = run_EAST_video(mmif, new_view)
        elif mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            mmif = run_EAST_image(mmif, new_view)
        return mmif


if __name__ == "__main__":
    td_tool = EAST_td()
    td_service = Restifier(td_tool)
    td_service.run()
