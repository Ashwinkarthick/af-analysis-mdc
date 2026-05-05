"""
Command-line interface for AF-Analysis
"""

import sys
import os
import argparse
import subprocess

from pathlib import Path


def main():
    """Launch the AF-Analysis Streamlit application"""
    parser = argparse.ArgumentParser(
        description="AF-Analysis - AlphaFold / AlphaPulldown Interaction Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  af-analysis
  af-analysis --directory /path/to/predictions
  af-analysis --directory /path/to/predictions --port 8502

""",
    )

    parser.add_argument(
        "--directory",
        type=str,
        default="",
        help="Path to directory containing AlphaPulldown predictions",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the server on (default: 8501)",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost",
        help="Server address (default: localhost)",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        help="Force opening a browser (disable headless mode).",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Force headless mode (never open a browser).",
    )

    args = parser.parse_args()

    if args.browser and args.no_browser:
        print("Error: use only one of --browser / --no-browser", file=sys.stderr)
        sys.exit(2)

    if not (1 <= args.port <= 65535):
        print(f"Error: invalid port {args.port}", file=sys.stderr)
        sys.exit(2)

    # Get the path to app.py
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Build streamlit command
    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.server_address,
        "--client.showSidebarNavigation",
        "false",
    ]

    # Handle browser settings
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    elif args.browser:
        cmd.extend(["--server.headless", "false"])

    # Add directory if provided
    if args.directory:
        if not Path(args.directory).exists():
            print(
                f"Error: Directory '{args.directory}' does not exist", file=sys.stderr
            )
            sys.exit(1)
        cmd.extend(["--", "--directory", args.directory])

    # Prepare environment
    env = os.environ.copy()
    env.setdefault("STREAMLIT_CLIENT_SHOW_SIDEBAR_NAVIGATION", "false")
    if args.directory:
        env["AF_ANALYSIS_DEFAULT_DIRECTORY"] = str(Path(args.directory).resolve())

    # Print startup message
    print("=" * 70)
    print("AF-Analysis - AlphaFold / AlphaPulldown Interaction Viewer")
    print("=" * 70)
    if args.directory:
        print(f"Directory: {args.directory}")
    print(f"Server: http://{args.server_address}:{args.port}")
    print("=" * 70)
    print("\nStarting server... Press Ctrl+C to stop")
    print()

    # Run streamlit
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running AF-Analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
