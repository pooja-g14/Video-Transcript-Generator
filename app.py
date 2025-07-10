from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import os
from main import process_video_for_transcript
from sqlalchemy.orm import Session
from database import get_db, Video

app = FastAPI(
    title="Video Transcription API",
    description="API for transcribing video files using Whisper and Silero VAD",
    version="1.0.0"
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload", 
    summary="Upload and transcribe a video file",
    response_description="Returns the transcription and saves it to database")
async def upload_file(video: UploadFile = File(...), db: Session = Depends(get_db)):
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    if not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    try:
        # Save uploaded file
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Process the video
        transcript = process_video_for_transcript(video_path)
        
        if transcript:
            # Save to database
            db_video = Video(filename=video.filename, transcript=transcript)
            db.add(db_video)
            db.commit()
            db.refresh(db_video)
            
            # Clean up video file
            os.remove(video_path)
            
            return JSONResponse(content={
                'message': 'Video processed successfully',
                'transcript': transcript,
                'video_id': db_video.id
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to process video")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcripts/{video_id}",
    summary="Get a transcript by video ID",
    response_description="Returns the transcript")
async def get_transcript(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return JSONResponse(content={
        'filename': video.filename,
        'transcript': video.transcript
    })

@app.get("/videos",
    summary="List all videos",
    response_description="Returns list of all videos and their transcripts")
async def list_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).all()
    return JSONResponse(content={
        'videos': [
            {
                'id': video.id,
                'filename': video.filename,
                'transcript': video.transcript
            } for video in videos
        ]
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)