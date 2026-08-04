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
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        # Starlette raises this if a disconnect message has already been
        # consumed during a close/reconnect race. Other RuntimeErrors remain
        # visible instead of being silently swallowed.
        if 'Cannot call "receive" once a disconnect message has been received.' not in str(exc):
            raise
    finally:
        manager.disconnect(websocket)
