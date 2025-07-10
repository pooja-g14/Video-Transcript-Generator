# Video-Transcript-Generator
Upload a video file, and generate a transcript of that video file.

- OpenAI's Whisper Model is used for transcription
- Fast API is used as the interface for uploading videos and getting the transcripts.
- PostgreSQL is used to store the video name and transcript.


To run this application:
1. Create a virtual environment (venv)
2. Activate the virtual environment (venv)
3. Install the libraries in requirements.txt
4. Create the database in PostgreSQL and add the Database URL to the .env file
5. In terminal, run: python app.py
6. Go to http://localhost:8000/docs to check the functioning of the app
