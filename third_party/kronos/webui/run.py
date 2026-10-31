#!/usr/bin/env python3
"""
Kronos Web UI startup script
"""

import os
import subprocess
import sys
import time
import webbrowser


def check_dependencies():
    """Check if dependencies are installed"""
    import importlib.util

    missing = [
        name
        for name in ("flask", "flask_cors", "numpy", "pandas", "plotly")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(f"❌ Missing dependency: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    print("✅ All dependencies installed")
    return True


def install_dependencies():
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Dependencies installation failed")
        return False


def main():
    """Main function"""
    print("🚀 Starting Kronos Web UI...")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == "y":
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return

    # Check model availability
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import importlib.util

        if importlib.util.find_spec("model") is None:
            raise ImportError("model")
        importlib.import_module("model")
        print("✅ Kronos model library available")
    except ImportError:
        print("⚠️  Kronos model library not available, will use simulated prediction")

    # Start Flask application
    print("\n🌐 Starting Web server...")

    # Set environment variables
    os.environ["FLASK_APP"] = "app.py"
    os.environ["FLASK_ENV"] = "development"

    # Start server
    try:
        from app import app

        print("✅ Web server started successfully!")
        print("🌐 Access URL: http://localhost:7070")
        print("💡 Tip: Press Ctrl+C to stop server")

        # Auto-open browser
        time.sleep(2)
        webbrowser.open("http://localhost:7070")

        # Start Flask application
        app.run(debug=True, host="0.0.0.0", port=7070)

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print("Please check if port 7070 is occupied")


if __name__ == "__main__":
    main()
