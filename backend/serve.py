import os

from waitress import serve

from app import app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    threads = int(os.environ.get("WAITRESS_THREADS", "8"))

    print("Starting production server with Waitress...")
    print(f"Serving on http://{host}:{port} (threads={threads})")

    serve(app, host=host, port=port, threads=threads)
