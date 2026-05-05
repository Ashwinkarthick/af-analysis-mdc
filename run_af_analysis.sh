#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export STREAMLIT_CLIENT_SHOW_SIDEBAR_NAVIGATION=false
exec streamlit run src/af_analysis/app.py --client.showSidebarNavigation=false "$@"
