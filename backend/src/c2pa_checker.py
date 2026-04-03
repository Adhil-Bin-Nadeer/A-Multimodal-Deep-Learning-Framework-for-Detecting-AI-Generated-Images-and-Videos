import json


try:
    import c2pa
    C2PA_AVAILABLE = True
    C2PA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on runtime package support
    c2pa = None
    C2PA_AVAILABLE = False
    C2PA_IMPORT_ERROR = str(exc)


def get_c2pa_runtime_status():
    return {
        "available": C2PA_AVAILABLE,
        "import_error": C2PA_IMPORT_ERROR,
        "has_reader": bool(getattr(c2pa, "Reader", None)) if C2PA_AVAILABLE else False,
        "has_c2pa_error": bool(getattr(c2pa, "C2paError", None)) if C2PA_AVAILABLE else False,
    }


def _parse_manifest_store(manifest_store):
    if not manifest_store:
        return None

    if isinstance(manifest_store, dict):
        return manifest_store

    if isinstance(manifest_store, str):
        return json.loads(manifest_store)

    return None


def _extract_ai_signal_from_assertions(assertions):
    if not isinstance(assertions, list):
        return False

    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue

        label = str(assertion.get("label", "")).lower()
        if "actions" not in label:
            continue

        action_payload = assertion.get("data", {}) or {}
        actions = action_payload.get("actions", [])
        if not isinstance(actions, list):
            continue

        for action in actions:
            if not isinstance(action, dict):
                continue
            source_type = str(action.get("digitalSourceType", "")).lower()
            if "trainedalgorithmicmedia" in source_type:
                return True

    return False

def check_c2pa(file_path):
    if not C2PA_AVAILABLE:
        return {
            "c2pa_present": False,
            "available": False,
            "message": "C2PA library not available in this runtime",
            "import_error": C2PA_IMPORT_ERROR,
        }

    c2pa_error_cls = getattr(c2pa, "C2paError", Exception)

    try:
        reader_cls = getattr(c2pa, "Reader", None)
        if reader_cls is None:
            return {
                "c2pa_present": False,
                "available": False,
                "message": "C2PA Reader API is not available in installed package",
            }

        reader = reader_cls(file_path)
        manifest_data = _parse_manifest_store(reader.json())

        if not manifest_data:
            return {
                "c2pa_present": False,
                "available": True,
                "message": "No C2PA manifest found",
            }

        active_manifest_label = manifest_data.get("active_manifest")
        if not active_manifest_label:
            return {
                "c2pa_present": False,
                "available": True,
                "message": "Manifest store exists but no active manifest",
            }

        manifests = manifest_data.get("manifests", {})
        active_manifest = manifests.get(active_manifest_label, {})

        validation_status = manifest_data.get("validation_status", [])
        is_valid = len(validation_status) == 0

        signature = active_manifest.get("signature_info", {})
        issuer = signature.get("issuer")

        assertions = active_manifest.get("assertions", [])
        is_ai = _extract_ai_signal_from_assertions(assertions)

        return {
            "c2pa_present": True,
            "available": True,
            "valid": is_valid,
            "issuer": issuer,
            "ai_generated": is_ai,
            "raw_data": manifest_data,
        }
    except c2pa_error_cls as exc:
        return {
            "c2pa_present": False,
            "available": True,
            "message": "C2PA read completed but no usable manifest was found",
            "error": str(exc),
        }
    except Exception as e:
        return {"c2pa_present": False, "available": True, "error": str(e)}
