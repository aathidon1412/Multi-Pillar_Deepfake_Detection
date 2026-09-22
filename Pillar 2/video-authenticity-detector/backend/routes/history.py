from fastapi import APIRouter, HTTPException
from backend.services.json_storage import list_results, delete_result, load_result

router = APIRouter(prefix="/api", tags=["History"])

@router.get("/history")
async def get_history():
    """
    Returns all past video authenticity detection records from JSON files.
    No SQL database required.
    """
    try:
        return list_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@router.delete("/history/{video_id}")
async def remove_history_item(video_id: str):
    """
    Deletes an analysis report and cleans up uploaded video and suspicious frames.
    """
    deleted = delete_result(video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Record '{video_id}' not found.")
    return {"message": f"Record '{video_id}' successfully deleted.", "video_id": video_id}
