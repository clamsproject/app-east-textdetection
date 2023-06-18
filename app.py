import argparse
from typing import Union

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier

from east_utils import *


class EAST_td(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        for videodocument in mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            # one view per video document
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            config = self.get_configuration(**parameters)
            mmif = run_EAST_video(mmif, videodocument, new_view, **config)
        if mmif.get_documents_by_type(DocumentTypes.ImageDocument):
            # one view for all image documents
            new_view = mmif.new_view()
            self.sign_view(new_view, parameters)
            mmif = run_EAST_image(mmif, new_view)
        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = EAST_td()

    http_app = Restifier(app, port=int(parsed_args.port)
    )
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
