#!/usr/bin/env bash
# Share the live portal on a temporary public link that switches itself off.
#
# Run INSIDE SageMaker (JupyterLab terminal), from the imyoo-portal folder:
#     bash share_live.sh        # 3 hours
#     bash share_live.sh 1.5    # or any number of hours
#
# Visitors need the password you type below. The Flare Analyst and "Explain my report" are turned off for them,
# so nobody can spend the AWS budget. Your own portal on port 8501 is untouched. Ctrl+C stops sharing early.
set -euo pipefail
HOURS="${1:-3}"
cd "$(dirname "$0")"

read -r -s -p "Choose a password for visitors: " PW; echo
[ "${#PW}" -ge 6 ] || { echo "Use at least 6 characters."; exit 1; }

mkdir -p ~/bin
if [ ! -x ~/bin/cloudflared ]; then
  echo "Downloading the tunnel tool (cloudflared)..."
  curl -fsSL -o ~/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x ~/bin/cloudflared
fi

PORTAL_PASSWORD="$PW" PORTAL_SHARED=1 streamlit run app.py --server.port 8502 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false \
  > /tmp/shared_portal.log 2>&1 &
PORTAL_PID=$!
trap 'kill "$PORTAL_PID" 2>/dev/null; echo; echo "Sharing stopped: the public link no longer works."' EXIT
sleep 6

END=$(date -d "+$(awk "BEGIN{print int($HOURS*60)}") minutes" "+%H:%M %Z" 2>/dev/null || echo "in $HOURS hours")
echo "Starting the public link (switches off automatically at $END)..."
timeout "${HOURS}h" ~/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8502 2>&1 \
  | tee /tmp/tunnel.log \
  | grep --line-buffered -o 'https://[a-z0-9-]*\.trycloudflare\.com' \
  | while read -r url; do echo; echo "  PUBLIC LINK: $url"; echo "  Share it with the password. Keep this terminal open."; echo; done || true
