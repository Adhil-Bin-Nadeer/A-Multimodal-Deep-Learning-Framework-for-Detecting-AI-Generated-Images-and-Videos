import json

# Try to import c2pa - may not be available on all platforms (e.g., Vercel)
try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

def check_c2pa(file_path):
    if not C2PA_AVAILABLE:
        return {
            "c2pa_present": False,
            "available": False,
            "message": "C2PA library not available on this platform"
        }
    
    try:
        # Read the manifest store from the file using Reader
        reader = c2pa.Reader(file_path)
        manifest_store_json = reader.json()

        if not manifest_store_json:
            return {"c2pa_present": False, "message": "No C2PA manifest found"}

        # Parse the JSON string returned by the library
        manifest_data = json.loads(manifest_store_json)
        
        # Get the active manifest
        active_manifest_label = manifest_data.get("active_manifest")
        if not active_manifest_label:
            return {"c2pa_present": False, "message": "Manifest store exists but no active manifest."}

        manifests = manifest_data.get("manifests", {})
        active_manifest = manifests.get(active_manifest_label, {})
        
        # Check validation status
        validation_status = manifest_data.get("validation_status", [])
        is_valid = len(validation_status) == 0  # Empty list means no errors

        # Extract signature info
        signature = active_manifest.get("signature_info", {})
        issuer = signature.get("issuer")
        
        # Look for assertions
        assertions = active_manifest.get("assertions", [])
        is_ai = False
        
        for assertion in assertions:
            if assertion.get("label") == "c2pa.actions":
                actions = assertion.get("data", {}).get("actions", [])
                for action in actions:
                    if "trainedAlgorithmicMedia" in action.get("digitalSourceType", ""):
                        is_ai = True

        return {
            "c2pa_present": True,
            "valid": is_valid,
            "issuer": issuer,
            "ai_generated": is_ai,
            "raw_data": manifest_data  # Return full data if needed
        }
    except c2pa.C2paError as e:
        # No manifest found or other C2PA-specific error
        return {"c2pa_present": False, "error": str(e)}
    except Exception as e:
        return {"c2pa_present": False, "error": str(e)}
