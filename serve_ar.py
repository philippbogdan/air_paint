#!/usr/bin/env python3
"""
Serve the latest drawing for AR viewing on iPhone.

Creates a local web server and displays a QR code.
Scan with iPhone camera → opens AR Quick Look automatically.

Usage:
    python serve_ar.py [session_folder]
"""

import argparse
import http.server
import os
import socket
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config.settings import get_output_dir


def get_local_ip() -> str:
    """Get the local IP address for LAN access."""
    try:
        # Connect to external address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_latest_session() -> Path:
    """Find the most recent session folder."""
    output_dir = get_output_dir()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    sessions = sorted(output_dir.glob("session_*"), reverse=True)
    if not sessions:
        raise FileNotFoundError("No sessions found. Draw something first!")

    return sessions[0]


def find_usdz_file(session_dir: Path) -> Path:
    """Find or create USDZ file in session."""
    usdz_path = session_dir / "drawing.usdz"

    if usdz_path.exists():
        return usdz_path

    # Try to convert from OBJ
    obj_path = session_dir / "drawing.obj"
    if obj_path.exists():
        print("USDZ not found, attempting conversion...")
        from export.usdz import convert_to_usdz
        result = convert_to_usdz(obj_path)
        if result and result.exists():
            return result

    raise FileNotFoundError(
        f"No USDZ file found and conversion failed.\n"
        f"Install Reality Converter from App Store and run:\n"
        f"  usdzconvert {obj_path} {usdz_path}"
    )


def generate_qr_terminal(url: str) -> str:
    """Generate QR code as ASCII art for terminal."""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Convert to ASCII
        lines = []
        for row in qr.modules:
            line = ""
            for cell in row:
                line += "██" if cell else "  "
            lines.append(line)
        return "\n".join(lines)
    except ImportError:
        return None


def generate_html_page(usdz_url: str, filename: str) -> str:
    """Generate HTML page with AR Quick Look link and QR code."""
    # Use a QR code API for the HTML page
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={usdz_url}"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AR View - {filename}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background: #1a1a1a;
            color: white;
        }}
        h1 {{ color: #00ff88; margin-bottom: 10px; }}
        .qr {{ margin: 30px 0; }}
        .qr img {{
            border-radius: 10px;
            background: white;
            padding: 10px;
        }}
        .ar-button {{
            display: inline-block;
            background: #00ff88;
            color: black;
            padding: 15px 40px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: bold;
            font-size: 18px;
            margin: 20px 0;
        }}
        .ar-button:hover {{ background: #00cc6a; }}
        .instructions {{
            background: #333;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: left;
        }}
        .instructions ol {{ margin: 10px 0; padding-left: 20px; }}
        .filename {{
            font-family: monospace;
            background: #333;
            padding: 5px 10px;
            border-radius: 5px;
        }}
        a {{ color: #00ff88; }}
    </style>
</head>
<body>
    <h1>3D Air Paint</h1>
    <p class="filename">{filename}</p>

    <div class="qr">
        <img src="{qr_url}" alt="QR Code">
    </div>

    <a href="{usdz_url}" rel="ar" class="ar-button">
        View in AR
    </a>

    <div class="instructions">
        <strong>On iPhone:</strong>
        <ol>
            <li>Scan the QR code with Camera app</li>
            <li>Tap the notification to open</li>
            <li>Tap "AR" to place in your room</li>
        </ol>
        <p><em>Or tap the button above if viewing on iPhone</em></p>
    </div>

    <p style="color: #666; font-size: 12px;">
        Direct link: <a href="{usdz_url}">{usdz_url}</a>
    </p>
</body>
</html>"""


class ARHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves USDZ with correct MIME type."""

    def __init__(self, *args, usdz_path: Path = None, **kwargs):
        self.usdz_path = usdz_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/":
            # Serve HTML page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            ip = get_local_ip()
            port = self.server.server_address[1]
            usdz_url = f"http://{ip}:{port}/drawing.usdz"
            html = generate_html_page(usdz_url, self.usdz_path.name)
            self.wfile.write(html.encode())

        elif self.path == "/drawing.usdz":
            # Serve USDZ file
            self.send_response(200)
            self.send_header("Content-type", "model/vnd.usdz+zip")
            self.send_header("Content-Disposition", f'inline; filename="{self.usdz_path.name}"')
            self.send_header("Content-Length", str(self.usdz_path.stat().st_size))
            self.end_headers()

            with open(self.usdz_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Quieter logging
        if "drawing.usdz" in str(args):
            print(f"  → iPhone downloaded the model!")


def serve_ar(session_path: Path = None, port: int = 8888, open_browser: bool = True):
    """Start AR server for the given session."""

    # Find session
    if session_path is None:
        session_path = find_latest_session()
    elif not session_path.is_absolute():
        session_path = get_output_dir() / session_path

    print(f"\nSession: {session_path.name}")

    # Find USDZ
    usdz_path = find_usdz_file(session_path)
    print(f"USDZ: {usdz_path.name} ({usdz_path.stat().st_size / 1024:.1f} KB)")

    # Get network info
    ip = get_local_ip()
    url = f"http://{ip}:{port}"
    usdz_url = f"{url}/drawing.usdz"

    # Create handler with USDZ path
    handler = lambda *args, **kwargs: ARHandler(*args, usdz_path=usdz_path, **kwargs)

    # Start server
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\n{'='*50}")
        print(f"AR Server running!")
        print(f"{'='*50}")
        print(f"\nOpen on iPhone: {url}")
        print(f"Direct USDZ:    {usdz_url}")

        # Show QR in terminal if possible
        qr_ascii = generate_qr_terminal(url)
        if qr_ascii:
            print(f"\nScan this QR code with iPhone:\n")
            print(qr_ascii)
        else:
            print("\n(Install 'qrcode' for terminal QR: pip install qrcode)")

        print(f"\n{'='*50}")
        print("Press Ctrl+C to stop")
        print(f"{'='*50}\n")

        # Open browser on Mac
        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Serve 3D drawing for AR viewing on iPhone"
    )
    parser.add_argument(
        "session",
        nargs="?",
        help="Session folder name (default: latest)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8888,
        help="Server port (default: 8888)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    args = parser.parse_args()

    session_path = Path(args.session) if args.session else None

    try:
        serve_ar(
            session_path=session_path,
            port=args.port,
            open_browser=not args.no_browser
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {args.port} is busy. Try: python serve_ar.py -p {args.port + 1}")
        else:
            raise


if __name__ == "__main__":
    main()
