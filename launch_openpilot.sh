#!/usr/bin/bash

[ ! -d "/data/community" ] && mkdir /data/community
[ ! -d "/data/community/logs" ] && mkdir /data/community/logs && echo "Created /data/community/logs"

DATE="date +%d-%m-%y--%I:%M%p"
LOG_DIR="/data/community/logs/$DATE"

export PASSIVE="0"
exec ./launch_chffrplus.sh 2>&1 | tee "$LOG_DIR"
