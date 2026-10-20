"""Image-count policy applies to the full history, with recovery after a refusal."""

import copy
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, serve_options
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler as mlx_handler
from tensorfold.server.prompts import prepare_images, prepare_prompt
from tensorfold.vision.images import ImageLimits
from tests.test_server_openai_compat import FakeApp, post_json
from tests.test_vision_server import Frontend, image_messages, prompt_app


@pytest.mark.parametrize("limit", [None, 8])
def test_mlx_app_count_reaches_request_preparation(limit, monkeypatch):
    from tensorfold.server.app import ChatApp
    from tensorfold.server.scheduler import Scheduler
    from tests.test_lane_server import FakeTokenizer

    # Preparation needs no engine rounds; don't start the model worker or watchdog.
    monkeypatch.setattr(Scheduler, "start", lambda self: None)
    app = ChatApp(NS(vision=Frontend()), FakeTokenizer(), served_name="fixture", checkpoint_slots=0,
                  engine_factory=lambda *args, **kwargs: NS(), vision_max_images=limit)
    accepted = 4 if limit is None else 8
    prepared = prepare_prompt(app, image_messages() * accepted, [], False, None, {})
    assert len(prepared.vision.image_hashes) == accepted
    with pytest.raises(RequestError, match=f"at most {accepted}"):
        prepare_prompt(app, image_messages() * (accepted + 1), [], False, None, {})


@pytest.mark.parametrize("bounds, message", [
    ({"max_total_encoded_bytes": 100}, "byte"),
    ({"max_total_pixels": 16}, "pixel"),
])
def test_higher_count_preserves_other_image_budgets(bounds, message):
    with pytest.raises(RequestError, match=message):
        prepare_images(Frontend(), image_messages() * 5, str, limits=ImageLimits(max_images=8, **bounds))


class ImageApp(FakeApp):
    """Keep MLX's real prompt preparation; replace only model/tokenizer work."""

    def __init__(self, limit):
        super().__init__()
        self.__dict__.update(vars(prompt_app(Frontend())))
        self.image_limits = ImageLimits(max_images=limit)

    def chat(self, messages, **kwargs):
        prepare_prompt(self, messages, kwargs.get("tools"), False, None, {})
        return super().chat(messages, **kwargs)


@pytest.mark.parametrize("backend", ["mlx"])
@pytest.mark.parametrize("value, vision, message", [("0", True, "positive integer"),
                                                   ("-1", True, "positive integer"),
                                                   ("8", False, "needs --vision")])
def test_bad_image_count_options_are_refused_before_checkpoint_reads(backend, value, vision, message, monkeypatch):
    from tensorfold import families

    monkeypatch.setattr(families, "read_config", lambda *args: pytest.fail("read checkpoint before refusal"))
    args = cli.build_parser().parse_args(["serve", "owner/model", "--vision-max-images", value,
                                         *(["--vision"] if vision else [])])
    with pytest.raises(ValueError, match=message):
        serve_options.check(args, NS(model_type="qwen3_5"), backend, "unused")

