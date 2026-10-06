import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.video_processor import save_uploaded_video, inspect_video
from backend.services.json_storage import update_status

router = APIRouter(prefix="/api", tags=["Upload"])

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts video upload, validates format & size, saves to storage/uploads/,
    and returns initial upload confirmation.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    try:
        video_id, saved_path, size_mb = await save_uploaded_video(file)
        
        # Verify video can be read by OpenCV
        try:
            video_details = inspect_video(saved_path)
        except Exception as e:
            # If invalid or unreadable, remove saved file
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file cannot be decoded as a valid video: {str(e)}"
            )

        # Set initial status
        update_status(video_id, "Uploaded", 0, "ready")

        return {
            "video_id": video_id,
            "filename": file.filename,
            "stored_filename": os.path.basename(saved_path),
            "status": "uploaded",
            "size_mb": size_mb,
            "video_details": video_details
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")
