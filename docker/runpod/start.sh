#!/usr/bin/env bash
set -euo pipefail

log() { echo "[start] $*"; }

start_nginx() {
  if command -v nginx >/dev/null 2>&1; then
    log "Starting nginx on :80"
    service nginx start || nginx
  else
    log "nginx not installed; skipping"
  fi
}

setup_ssh() {
  log "Configuring SSH"
  mkdir -p /var/run/sshd /root/.ssh
  chmod 700 /root/.ssh
  # Host keys
  if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
  fi
  # Authorized keys from env
  if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
  fi
  # Start SSHD
  /usr/sbin/sshd
  log "SSH running. Example: ssh -i ~/.ssh/id_rsa -p 22 root@<host>"
}

start_jupyter() {
  if command -v jupyter >/dev/null 2>&1; then
    local token_flag=""
    if [ -n "${JUPYTER_PASSWORD:-}" ]; then
      token_flag="--ServerApp.token=${JUPYTER_PASSWORD} --ServerApp.password=''"
    fi
    log "Starting JupyterLab on :9999"
    nohup jupyter lab --no-browser --ip=0.0.0.0 --port=9999 $token_flag >/var/log/jupyter.log 2>&1 &
  else
    log "Jupyter not installed; skipping"
  fi
}

start_filebrowser() {
  if command -v filebrowser >/dev/null 2>&1; then
    log "Starting FileBrowser on :9090"
    nohup filebrowser -r / -p 9090 >/var/log/filebrowser.log 2>&1 &
  else
    log "FileBrowser not installed; skipping"
  fi
}

export_env() {
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
}

start_nginx
setup_ssh
start_jupyter
start_filebrowser
export_env

log "Container started. Services: SSH:22, Nginx:80, FileBrowser:9090, Jupyter:9999"
log "Tail logs: tail -f /var/log/jupyter.log /var/log/filebrowser.log"

sleep infinity
