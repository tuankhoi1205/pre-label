import json
import os
from pathlib import Path
import sys

# Use the versioned project source mounted alongside dataset/model files.
if Path('/workspace/src/prelabel').is_dir():
    sys.path.insert(0, '/workspace/src')

from prelabel.config import load_config
from prelabel.serverless import CenterPointFunction


def init_context(context):
    cfg = load_config(os.environ.get("PRELABEL_CONFIG", "/workspace/configs/mini.yaml"))
    cfg["run_dir"] = os.environ.get("PRELABEL_RUN_DIR", "/workspace/runs/ui-centerpoint")
    # Registration is possible before the dataset download finishes. Validate
    # the prepared inputs and initialize the model on the first actual request.
    context.user_data.cfg = cfg
    context.user_data.model = None


def handler(context, event):
    try:
        payload = json.loads(event.body) if isinstance(event.body, (str, bytes, bytearray)) else event.body
        if context.user_data.model is None:
            context.user_data.model = CenterPointFunction(context.user_data.cfg)
        result = (context.user_data.model.validate(payload) if isinstance(payload, dict)
                  and payload.get('validate_only') is True else context.user_data.model.predict(payload))
        return context.Response(body=json.dumps(result, allow_nan=False),
            content_type="application/json", status_code=200)
    except (ValueError, json.JSONDecodeError) as exc:
        return context.Response(body=json.dumps({"error": str(exc)}),
            content_type="application/json", status_code=400)
    except FileNotFoundError:
        return context.Response(body=json.dumps({"error": "Prepare the nuScenes run before clicking Annotate: dataset and manifest are required."}),
            content_type="application/json", status_code=400)
    except (RuntimeError, OSError) as exc:
        context.logger.error("CenterPoint inference failed", error=str(exc))
        return context.Response(body=json.dumps({"error": str(exc)}),
            content_type="application/json", status_code=500)
