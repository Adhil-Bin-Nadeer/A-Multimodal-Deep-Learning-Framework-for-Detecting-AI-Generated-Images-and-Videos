from datetime import datetime


def generate_forensic_report(analysis_result: dict) -> dict:
    """
    Takes the raw analysis result from the detection pipeline and generates
    a structured forensic report.
    
    Args:
        analysis_result: Dictionary containing:
            - filename: Name of the analyzed file
            - layers: Dict with c2pa, synthid, ai_model results
            - final_verdict: "AI Image" or "Real Image"
            - confidence: Confidence percentage
            - is_ai_generated: Boolean flag
    
    Returns:
        Dictionary containing the forensic report
    """
    
    forensic_summary = build_image_forensic_summary(analysis_result)
    report = _generate_forensic_report(analysis_result, forensic_summary)
    
    return {
        "success": True,
        "enhanced_report": report,
        "raw_analysis": analysis_result,
        "summary": _extract_summary(analysis_result),
        "forensic_summary": forensic_summary,
    }


def _safe_percent(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _image_module_summary(module_name: str, module_data: dict) -> str:
    if not isinstance(module_data, dict):
        return f"{module_name} check was not available."

    if module_data.get('available') is False:
        return f"{module_name} check was unavailable."

    detected = bool(module_data.get('detected'))
    score = _safe_percent(module_data.get('ai_score', 0.0)) * 100.0

    if module_name == "Frequency":
        if detected:
            return f"Frequency analysis found checkerboard-style peaks linked to synthetic up-sampling ({score:.1f}% signal strength)."
        return f"Frequency analysis did not find strong checkerboard up-sampling artifacts ({score:.1f}% signal strength)."

    if module_name == "Anatomy":
        if detected:
            return f"Pose and hand landmark analysis found anatomical inconsistencies ({score:.1f}% anomaly strength)."
        return f"Pose and hand landmark analysis did not find strong anatomical inconsistencies ({score:.1f}% anomaly strength)."

    if module_name == "Entropy":
        if detected:
            return f"Local entropy analysis found uneven texture/detail distribution between regions ({score:.1f}% anomaly strength)."
        return f"Local entropy analysis looked consistent with natural scene detail ({score:.1f}% anomaly strength)."

    return f"{module_name} check completed."


def build_image_forensic_summary(analysis_result: dict) -> dict:
    layers = analysis_result.get('layers', {}) or {}
    verdict = analysis_result.get('final_verdict', 'Unknown')
    confidence = _safe_percent(analysis_result.get('confidence', 0.0))
    is_ai = bool(analysis_result.get('is_ai_generated', False))

    c2pa = layers.get('c2pa', {}) or {}
    synthid = layers.get('synthid', {}) or {}
    ai_model = layers.get('ai_model', {}) or {}
    forensic_modules = ai_model.get('forensic_modules', {}) if isinstance(ai_model, dict) else {}

    method = _get_detection_method(layers)
    points = []

    points.append(f"Primary evidence came from {method} with an overall confidence of {confidence:.1f}%.")

    if c2pa.get('c2pa_present') and c2pa.get('ai_generated'):
        points.append("Signed C2PA metadata explicitly declared AI generation.")
    elif c2pa.get('c2pa_present'):
        points.append("Signed C2PA metadata was present, and by current policy this is treated as AI-generated content.")
    elif c2pa.get('status') == 'unavailable' or c2pa.get('available') is False:
        points.append("C2PA provenance data could not be checked in this environment.")
    else:
        points.append("No signed C2PA provenance metadata was found in the file.")

    if synthid.get('status') == 'complete':
        if synthid.get('is_watermarked'):
            points.append(
                f"SynthID watermark detection was positive at {_safe_percent(synthid.get('confidence', 0.0)):.1f}% confidence."
            )
        else:
            points.append(
                f"No SynthID watermark was detected (detector confidence {_safe_percent(synthid.get('confidence', 0.0)):.1f}% for this negative result)."
            )
    elif synthid.get('status') == 'skipped':
        points.append("SynthID analysis was skipped because earlier signed evidence was already sufficient.")
    elif synthid:
        points.append("SynthID analysis did not provide a decisive watermark result.")

    if ai_model.get('status') == 'complete':
        pipeline_override = ai_model.get('pipeline_override', {}) if isinstance(ai_model, dict) else {}
        label = ai_model.get('label', 'Unknown')
        model_confidence = _safe_percent(ai_model.get('confidence', 0.0))
        if pipeline_override.get('applied'):
            points.append(
                "The AI image model produced a low-confidence AI result, but the pipeline overrode it to real "
                "because SynthID was negative and forensic support was weak."
            )
        else:
            points.append(f"The AI image model classified the file as {label} with {model_confidence:.1f}% confidence.")

        frequency = forensic_modules.get('frequency', {})
        anatomy = forensic_modules.get('anatomy', {})
        entropy = forensic_modules.get('entropy', {})
        module_points = [
            _image_module_summary("Frequency", frequency),
            _image_module_summary("Anatomy", anatomy),
            _image_module_summary("Entropy", entropy),
        ]
        points.extend(module_points)
    elif ai_model.get('status') == 'skipped':
        points.append("The AI image model was skipped because earlier evidence already supported the decision.")
    else:
        points.append("The AI image model was not available for this file.")

    conclusion = (
        f"The final decision is {verdict}."
        if is_ai else
        f"The final decision is {verdict}, meaning the file appears authentic under the current checks."
    )

    return {
        "title": "Image Forensic Summary",
        "method": method,
        "confidence": confidence,
        "points": points[:7],
        "conclusion": conclusion,
    }


def build_video_forensic_summary(video_result: dict) -> dict:
    label = str(video_result.get('label', 'UNKNOWN')).upper()
    confidence = _safe_percent(video_result.get('confidence', 0.0))
    source = video_result.get('source', 'Video AI Detector')
    explainability = video_result.get('explainability', {}) or {}
    metrics = explainability.get('metrics', {}) if isinstance(explainability, dict) else {}
    frames = explainability.get('frames', []) if isinstance(explainability, dict) else []
    frame_scores = [_safe_percent(frame.get('ai_probability', 0.0)) for frame in frames if isinstance(frame, dict)]

    frames_analyzed = int(metrics.get('frames_analyzed', len(frames) or 0))
    flagged_frames = int(metrics.get('flagged_frames', sum(score >= 50.0 for score in frame_scores)))
    average_score = metrics.get('average_ai_probability')
    peak_score = metrics.get('peak_ai_probability')

    if average_score is None and frame_scores:
        average_score = sum(frame_scores) / len(frame_scores)
    if peak_score is None and frame_scores:
        peak_score = max(frame_scores)

    points = []
    points.append("The video result was estimated from sampled frames taken across the clip.")
    if frames_analyzed:
        points.append(f"{frames_analyzed} frames were analyzed for AI-like visual patterns.")
    else:
        artifacts = explainability.get('artifacts', []) if isinstance(explainability, dict) else []
        for item in artifacts:
            if isinstance(item, str) and item.startswith("Frames analyzed:"):
                points.append(item + ".")
                break

    if frames_analyzed and average_score is not None and peak_score is not None:
        points.append(f"{flagged_frames} of {frames_analyzed} sampled frames crossed the AI decision threshold.")
        points.append(f"Average sampled-frame AI score was {_safe_percent(average_score):.1f}%, with a peak frame score of {_safe_percent(peak_score):.1f}%.")

    points.append(f"The final video decision was {label} with {confidence:.1f}% confidence.")

    return {
        "title": "Video Forensic Summary",
        "method": source,
        "confidence": confidence,
        "points": points[:5],
        "conclusion": "The video verdict is based on agreement across the sampled frames rather than on a single frame alone.",
    }


def _generate_forensic_report(analysis_result: dict, forensic_summary: dict) -> str:
    """Generate a short, presentation-friendly forensic report."""

    filename = analysis_result.get('filename', 'Unknown')
    verdict = analysis_result.get('final_verdict', 'Unknown')
    confidence = _safe_percent(analysis_result.get('confidence', 0))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    point_lines = "\n".join(f"- {item}" for item in forensic_summary.get('points', []))

    return f"""# FORENSIC SUMMARY

## Result
- File: {filename}
- Verdict: {verdict}
- Confidence: {confidence:.1f}%
- Time: {timestamp}
- Main Method: {forensic_summary.get('method', 'Unknown')}

## Key Findings
{point_lines}

## Conclusion
{forensic_summary.get('conclusion', 'The report summarizes the main detection evidence.')}
"""


def _extract_summary(analysis_result: dict) -> dict:
    """Extract a concise summary from the analysis."""
    
    return {
        "verdict": analysis_result.get('final_verdict', 'Unknown'),
        "confidence": analysis_result.get('confidence', 0),
        "is_ai_generated": analysis_result.get('is_ai_generated', False),
        "detection_method": _get_detection_method(analysis_result.get('layers', {}))
    }


def _get_detection_method(layers: dict) -> str:
    """Determine which detection method provided the result."""
    
    c2pa = layers.get('c2pa', {})
    if c2pa.get('c2pa_present'):
        if c2pa.get('ai_generated'):
            return "C2PA Manifest AI Declaration"
        return "C2PA Metadata Presence Policy"

    synthid = layers.get('synthid', {})
    if synthid.get('status') == 'complete' and synthid.get('is_watermarked'):
        return "SynthID Watermark Detection"
    
    ai_model = layers.get('ai_model', {})
    if ai_model.get('status') == 'complete':
        return "AI Detection Model (ResNet + ViT Ensemble)"

    return "Unknown"


# -------- CLI TESTING --------
if __name__ == "__main__":
    sample_result = {
        "success": True,
        "filename": "test_image.jpg",
        "layers": {
            "c2pa": {"c2pa_present": False, "message": "No C2PA manifest found"},
            "synthid": {"status": "complete", "is_watermarked": False, "confidence": 12.5, "message": "No SynthID watermark detected"},
            "ai_model": {
                "status": "complete",
                "label": "AI Generated",
                "ai_probability": 0.95,
                "confidence": 95.0
            }
        },
        "final_verdict": "AI Generated",
        "confidence": 95.0,
        "is_ai_generated": True
    }
    
    print("Testing Forensic Report Generation...")
    print("=" * 50)
    
    report = generate_forensic_report(sample_result)
    
    if report.get("success"):
        print(report.get("enhanced_report"))
    else:
        print(f"Error: {report.get('error')}")
        print("\nBasic Report:")
        print(report.get("basic_report"))
