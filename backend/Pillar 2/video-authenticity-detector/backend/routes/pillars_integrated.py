import os
import io
import sys
import uuid
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from backend.services.json_storage import save_result

# Root repository directory
BASE_REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_REPO_DIR))

router = APIRouter(prefix="/api/pillars", tags=["Integrated Pillars"])

# Cache variables for heavy models
_cached_p1 = None
_cached_p5 = None

def get_p1_model():
    global _cached_p1
    if _cached_p1 is None:
        from core import load_pillar1_vit
        _cached_p1 = load_pillar1_vit()
    return _cached_p1

def get_p5_model():
    global _cached_p5
    if _cached_p5 is None:
        from core import load_pillar5_ml_bundle
        _cached_p5 = load_pillar5_ml_bundle()
    return _cached_p5


# ----------------- UNIVERSAL MODALITY ROUTER & CONSENSUS -----------------
@router.post("/universal")
async def analyze_universal_media(
    file: UploadFile = File(...),
    p4_benford_enabled: bool = Form(False),
    audio_mode: str = Form("spoken")
):
    """
    Universal multi-pillar media pipeline:
    Auto-detects modality (Image, Video, Audio) and executes all applicable analytical pillars
    with unified consensus decision fusion.
    """
    filename = file.filename or "media.tmp"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
    VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
    AUDIO_EXTS = {"wav", "mp3", "flac", "ogg", "m4a"}

    content = await file.read()

    # Case A: Video
    if ext in VIDEO_EXTS:
        return {
            "modality": "video",
            "redirect_endpoint": "/api/upload",
            "message": "Use /api/upload and /api/analyze for asynchronous video forensic telemetry.",
            "filename": filename
        }

    # Case B: Image -> Pillar 1 (ViT) + Pillar 5 (Shadow RANSAC) + Pillar 4 (Benford OCR) + Consensus
    if ext in IMAGE_EXTS:
        try:
            from core import run_pillar1_inference, run_pillar4_inference, run_pillar5_inference, fuse_multi_pillar_verdict, synthesize_multi_pillar_xai
            image_to_process = Image.open(io.BytesIO(content)).convert("RGB")
            img_np = np.array(image_to_process)

            processor_or_tfm, p1_model, device, p1_model_name = get_p1_model()
            p5_bundle, p5_model_filename = get_p5_model()

            p1_res = run_pillar1_inference(
                image_to_process, processor_or_tfm, p1_model, device, model_name=p1_model_name
            )
            p5_res = run_pillar5_inference(img_np, pil_img=image_to_process, p5_bundle=p5_bundle)

            p4_res = run_pillar4_inference(img_np, is_pdf=False) if p4_benford_enabled else {
                "applicable": False, "verdict": "DISABLED", "confidence": 0, "digits_count": 0, "reason": "Pillar 4 disabled."
            }

            fusion_res = fuse_multi_pillar_verdict(
                image_to_process, img_np, p1_res, p4_res, p5_res
            )

            # Strip non-JSON serializable numpy array / PIL objects
            clean_p5 = {k: v for k, v in p5_res.items() if k not in ("overlay",)}
            clean_p4 = {k: v for k, v in p4_res.items() if k not in ("extracted_image",)}

            record_id = f"IMG_{uuid.uuid4().hex[:6].upper()}"
            file_size_mb = round(len(content) / (1024 * 1024), 2)
            now_iso = datetime.now().isoformat()
            stored_filename = f"{record_id}.{ext}"

            # Save uploaded original file to uploads/ for static serving in ResultPage
            upload_dir = os.path.join(BASE_REPO_DIR, "Pillar 2", "video-authenticity-detector", "storage", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            stored_file_path = os.path.join(upload_dir, stored_filename)
            try:
                with open(stored_file_path, "wb") as f:
                    f.write(content)
            except Exception as fe:
                print(f"[WARN] Failed to write uploaded image to storage: {fe}")

            raw_verdict = fusion_res["verdict"]
            pred = "AI_GENERATED" if ("FAKE" in raw_verdict or "DEEPFAKE" in raw_verdict or "SYNTHETIC" in raw_verdict or not fusion_res.get("is_real")) else "REAL"

            p1_xai = p1_res.get("xai")
            p4_xai = p4_res.get("xai")
            p5_xai = p5_res.get("xai")
            consensus_xai = fusion_res.get("xai")

            combined_xai = synthesize_multi_pillar_xai(
                pillar1=p1_xai,
                pillar4=p4_xai,
                pillar5=p5_xai,
                consensus=consensus_xai
            )

            report = {
                "video_id": record_id,
                "modality": "image",
                "filename": filename,
                "file": {
                    "original_filename": filename,
                    "stored_filename": stored_filename,
                    "format": ext,
                    "size_mb": file_size_mb
                },
                "classification": {
                    "prediction": pred,
                    "confidence": round(float(fusion_res["confidence"]) / (100.0 if fusion_res["confidence"] > 1.0 else 1.0), 2),
                    "scores": {
                        "real": round(float(fusion_res["confidence"]) / 100.0, 2) if fusion_res["is_real"] else round(1.0 - (float(fusion_res["confidence"]) / 100.0), 2),
                        "ai_generated": round(1.0 - (float(fusion_res["confidence"]) / 100.0), 2) if fusion_res["is_real"] else round(float(fusion_res["confidence"]) / 100.0, 2),
                        "forged": 0.05
                    }
                },
                "consensus": {
                    "verdict": fusion_res["verdict"],
                    "confidence": round(float(fusion_res["confidence"]), 2),
                    "is_real": bool(fusion_res["is_real"]),
                    "override_reason": fusion_res.get("override_reason"),
                    "engines": f"{p1_model_name} + {p5_model_filename}"
                },
                "pillar1": p1_res,
                "pillar5": clean_p5,
                "pillar4": clean_p4,
                "xai": combined_xai,
                "processing": {
                    "status": "completed",
                    "processed_at": now_iso
                }
            }

            try:
                save_result(record_id, report)
            except Exception as se:
                print(f"[WARN] Failed to write image result to storage: {se}")

            return report
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image universal pipeline error: {str(e)}")

    # Case C: Audio -> Pillar 3 (Wav2Vec2 / Acoustic Transformer + Demixing)
    if ext in AUDIO_EXTS:
        from core import classify_audio, synthesize_multi_pillar_xai
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            internal_mode = "music" if "music" in audio_mode.lower() or "song" in audio_mode.lower() else "spoken"
            results = classify_audio(tmp_path, mode=internal_mode)
            is_fake = results["prediction"] == "FAKE"

            record_id = f"AUD_{uuid.uuid4().hex[:6].upper()}"
            file_size_mb = round(len(content) / (1024 * 1024), 2)
            now_iso = datetime.now().isoformat()
            pred = "AI_GENERATED" if is_fake else "REAL"
            conf = round(float(results["confidence"]) / (100.0 if results["confidence"] > 1.0 else 1.0), 2)
            stored_filename = f"{record_id}.{ext}"

            # Save uploaded audio file to storage/uploads for web playback
            upload_dir = Path(__file__).resolve().parent.parent / "storage" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            stored_file_path = upload_dir / stored_filename
            try:
                with open(stored_file_path, "wb") as f:
                    f.write(content)
            except Exception as fe:
                print(f"[WARN] Failed to write uploaded audio to storage: {fe}")

            p3_xai = results.get("xai")
            combined_xai = synthesize_multi_pillar_xai(pillar3=p3_xai)

            is_loc = results.get("is_localized_tamper", False)
            if is_loc:
                consensus_verdict = "LOCALIZED VOICE TAMPERING / SPLICED"
                pred = "AI_GENERATED"
            elif is_fake:
                consensus_verdict = "AI SYNTHESIZED VOICE"
            else:
                consensus_verdict = "AUTHENTIC HUMAN VOICE"

            report = {
                "video_id": record_id,
                "modality": "audio",
                "filename": filename,
                "file": {
                    "original_filename": filename,
                    "stored_filename": stored_filename,
                    "format": ext,
                    "size_mb": file_size_mb
                },
                "classification": {
                    "prediction": pred,
                    "confidence": conf,
                    "scores": {
                        "real": round(float(results.get("real_confidence", 0.0)) / (100.0 if results.get("real_confidence", 0.0) > 1.0 else 1.0), 2),
                        "ai_generated": round(float(results.get("fake_confidence", 0.0)) / (100.0 if results.get("fake_confidence", 0.0) > 1.0 else 1.0), 2),
                        "forged": 0.85 if is_loc else 0.0
                    }
                },
                "consensus": {
                    "verdict": consensus_verdict,
                    "confidence": round(float(results["confidence"]), 2),
                    "is_real": not is_fake,
                    "engines": "Wav2Vec2 / Acoustic Transformer + HPSS Demixing"
                },
                "pillar3": results,
                "xai": combined_xai,
                "processing": {
                    "status": "completed",
                    "processed_at": now_iso
                }
            }

            try:
                save_result(record_id, report)
            except Exception as se:
                print(f"[WARN] Failed to write audio result to storage: {se}")

            return report
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Audio universal pipeline error: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Case D: PDF / Document -> Pillar 4
    if ext in PDF_EXTS:
        from core import run_pillar4_inference, synthesize_multi_pillar_xai
        p4_res = run_pillar4_inference(content, is_pdf=True)
        raw_v = p4_res.get("verdict", "")
        is_fake = "FORGED" in raw_v or "MANIPULATED" in raw_v

        record_id = f"DOC_{uuid.uuid4().hex[:6].upper()}"
        file_size_mb = round(len(content) / (1024 * 1024), 2)
        now_iso = datetime.now().isoformat()
        pred = "FORGED" if is_fake else ("REAL" if "AUTHENTIC" in raw_v else "INCONCLUSIVE")
        raw_conf = float(p4_res.get("confidence", 85.0))
        conf = round(raw_conf / (100.0 if raw_conf > 1.0 else 1.0), 2)

        p4_xai = p4_res.get("xai")
        combined_xai = synthesize_multi_pillar_xai(pillar4=p4_xai)

        report = {
            "video_id": record_id,
            "modality": "pdf",
            "filename": filename,
            "file": {
                "original_filename": filename,
                "stored_filename": filename,
                "format": ext,
                "size_mb": file_size_mb
            },
            "classification": {
                "prediction": pred,
                "confidence": conf,
                "scores": {
                    "real": round(1.0 - conf, 2) if is_fake else conf,
                    "ai_generated": 0.05,
                    "forged": conf if is_fake else round(1.0 - conf, 2)
                }
            },
            "consensus": {
                "verdict": p4_res.get("verdict", "INCONCLUSIVE"),
                "confidence": round(float(p4_res.get("confidence", 0.0)), 2),
                "is_real": not is_fake,
                "engines": "Statistical Benford OCR & Revision Forensics"
            },
            "pillar4": p4_res,
            "xai": combined_xai,
            "processing": {
                "status": "completed",
                "processed_at": now_iso
            }
        }

        try:
            save_result(record_id, report)
        except Exception as se:
            print(f"[WARN] Failed to write pdf result to storage: {se}")

        return report

    raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")


# ----------------- PILLAR 3: AUDIO DETECTION -----------------
@router.post("/audio")
async def analyze_pillar3_audio(
    file: UploadFile = File(...),
    mode: str = Form("spoken")
):
    """
    Pillar 3: Audio & Speech Deepfake Detection using core.classify_audio
    """
    from core import classify_audio, synthesize_multi_pillar_xai
    temp_ext = os.path.splitext(file.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        internal_mode = "music" if "music" in mode.lower() or "song" in mode.lower() else "spoken"
        results = classify_audio(tmp_path, mode=internal_mode)
        is_fake = results["prediction"] == "FAKE"
        p3_xai = results.get("xai")
        xai_bundle = synthesize_multi_pillar_xai(pillar3=p3_xai)

        return {
            "prediction": results["prediction"],
            "verdict": "AI SYNTHESIZED VOICE" if is_fake else "AUTHENTIC HUMAN VOICE",
            "is_real": not is_fake,
            "confidence": round(float(results["confidence"]), 2),
            "real_confidence": round(float(results["real_confidence"]), 2),
            "fake_confidence": round(float(results["fake_confidence"]), 2),
            "is_music": bool(results.get("is_music", False)),
            "is_demixed": bool(results.get("is_demixed", False)),
            "duration": round(float(results.get("duration", 0.0)), 2),
            "samplerate": results.get("samplerate", 16000),
            "mode": mode,
            "filename": file.filename,
            "pillar3": results,
            "xai": xai_bundle
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ----------------- PILLAR 4: DOCUMENT & BENFORD FORENSICS -----------------
@router.post("/document")
async def analyze_pillar4_document(file: UploadFile = File(...)):
    """
    Pillar 4: Document, Invoice & PDF Statistical Forensics using core.run_pillar4_inference
    """
    from core import run_pillar4_inference, synthesize_multi_pillar_xai
    content = await file.read()
    filename = file.filename or "document.png"
    is_pdf = filename.lower().endswith(".pdf")

    try:
        if is_pdf:
            res = run_pillar4_inference(content, is_pdf=True)
        else:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            res = run_pillar4_inference(np.array(pil_img), is_pdf=False)

        is_fake = "FORGED" in res.get("verdict", "")
        p4_xai = res.get("xai")
        xai_bundle = synthesize_multi_pillar_xai(pillar4=p4_xai)

        return {
            "applicable": res.get("applicable", True),
            "verdict": res.get("verdict", "INCONCLUSIVE"),
            "is_real": not is_fake,
            "confidence": round(float(res.get("confidence", 85.0)), 2),
            "digits_count": res.get("digits_count", 0),
            "mae": round(float(res.get("mae", 0.0)), 4) if "mae" in res else None,
            "chi_square": round(float(res.get("chi_square", 0.0)), 2) if "chi_square" in res else None,
            "p_value": round(float(res.get("p_value", 0.0)), 4) if "p_value" in res else None,
            "obs_freqs": res.get("obs_freqs", {}),
            "expected_freqs": res.get("expected_freqs", {}),
            "reason": res.get("reason", ""),
            "filename": filename,
            "pillar4": res,
            "xai": xai_bundle
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis error: {str(e)}")


# ----------------- PILLARS 1 & 5: DUAL IMAGE FORENSICS -----------------
@router.post("/image")
async def analyze_pillar1_and_5_image(file: UploadFile = File(...)):
    """
    Pillars 1 & 5: ViT Neural Spectra + Shadow RANSAC Physics with Consensus Fusion
    """
    from core import run_pillar1_inference, run_pillar4_inference, run_pillar5_inference, fuse_multi_pillar_verdict, synthesize_multi_pillar_xai
    content = await file.read()
    try:
        image_to_process = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(image_to_process)

        processor_or_tfm, p1_model, device, p1_model_name = get_p1_model()
        p5_bundle, p5_model_filename = get_p5_model()

        p1_res = run_pillar1_inference(
            image_to_process, processor_or_tfm, p1_model, device, model_name=p1_model_name
        )
        p5_res = run_pillar5_inference(img_np, pil_img=image_to_process, p5_bundle=p5_bundle)
        p4_res = run_pillar4_inference(img_np, is_pdf=False)

        fusion_res = fuse_multi_pillar_verdict(
            image_to_process, img_np, p1_res, p4_res, p5_res
        )

        # Strip non-JSON serializable numpy array / PIL objects
        clean_p5 = {k: v for k, v in p5_res.items() if k not in ("overlay",)}
        clean_p4 = {k: v for k, v in p4_res.items() if k not in ("extracted_image",)}

        p1_xai = p1_res.get("xai")
        p4_xai = p4_res.get("xai")
        p5_xai = p5_res.get("xai")
        consensus_xai = fusion_res.get("xai")

        combined_xai = synthesize_multi_pillar_xai(
            pillar1=p1_xai,
            pillar4=p4_xai,
            pillar5=p5_xai,
            consensus=consensus_xai
        )

        return {
            "verdict": fusion_res["verdict"],
            "is_real": bool(fusion_res["is_real"]),
            "confidence": round(float(fusion_res["confidence"]), 2),
            "override_reason": fusion_res.get("override_reason"),
            "engines": f"{p1_model_name} + {p5_model_filename}",
            "pillar1": p1_res,
            "pillar5": clean_p5,
            "pillar4": clean_p4,
            "xai": combined_xai,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")


