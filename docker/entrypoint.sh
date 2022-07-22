#!/bin/sh
Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset &
export DISPLAY=:99
exec "$@"