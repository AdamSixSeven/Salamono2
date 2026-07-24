from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    manager = websocket.app.state.ws_manager
    camera_id = (websocket.query_params.get("camera_id") or "").strip() or None
    include_frame = (
        websocket.query_params.get("include_frame", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    binary_frames = (
        websocket.query_params.get("binary", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    await manager.connect(
        websocket,
        camera_id=camera_id,
        include_frame=include_frame,
        binary_frames=binary_frames,
    )
    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
