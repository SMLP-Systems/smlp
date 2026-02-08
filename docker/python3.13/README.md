# Python 3.13 Development Container with GUI Support

Docker image with Python 3.13 and GUI application support via VNC.

## Overview

This container provides a complete Python 3.13 development environment with support for running GUI applications through VNC. It's portable across different systems including native Linux, WSL2, and Docker Desktop.

**Image Details:**
- **Docker Hub:** `mdmitry1/python313-dev:latest`
- **Disk Size:** ~8GB (extracted), ~2GB (compressed)
- **Base:** Ubuntu 24.04 with Python 3.13

## Quick Start

### Pull the Image

```bash
docker pull mdmitry1/python313-dev:latest
```

### Run Container with VNC

```bash
docker run -it -p 5900:5900 mdmitry1/python313-dev:latest
```

## Setting Up GUI Support using for X11 using socat

1. Install `socat`, if it is not installed

```
sudo apt install socat
```

2. Run command

```bash
pkill socat
```

3. Run command

```bash
enter_released_container_x11_forwarding
```

## Setting Up GUI Support using for WSL2 with wslg installed

```
enter_released_container_wslg
```

## Setting Up GUI Support using VNC

Once inside the container, set up the virtual display and VNC server:

```bash
# 1. Start virtual X server
Xvfb :99 -screen 0 1280x1024x24 &

# 2. Set display environment variable
export DISPLAY=:99

# 3. Start VNC server
x11vnc -display :99 -forever -nopw -shared -rfbport 5900

# 4. Enable clipboard sync (optional but recommended)
apt-get update && apt-get install -y autocutsel
autocutsel -selection CLIPBOARD -fork
autocutsel -selection PRIMARY -fork
```

## Connecting to VNC

### Using Remmina (Recommended for Linux)

```bash
# Install Remmina
sudo apt-get install remmina remmina-plugin-vnc

# Launch and connect
remmina
# Create connection to: localhost:5900
```

## Clipboard Support

### Host System Setup (Linux)

For bidirectional clipboard sync between host and VNC, install autocutsel on your host:

```bash
# Install autocutsel
sudo apt-get install autocutsel

# Add to ~/.xprofile
autocutsel -selection CLIPBOARD -fork
autocutsel -selection PRIMARY -fork
```

### In Container

Already covered in the setup steps above. Use `autocutsel` to sync clipboard selections.

### Clipboard Usage in xterm

- **Paste from host:** Press `Shift+Insert` in xterm
- **Copy to host:** Select text in xterm (auto-copies to PRIMARY)

### Clipboard with gvim (More Reliable)

gvim handles clipboard better than xterm:
- Visual select + `"+y` - copy to CLIPBOARD
- `"+p` - paste from CLIPBOARD
- Or use visual mode + right-click menu

## File Operations

### Copy Files to Container

```bash
# Find your container name/ID
docker ps

# Copy file from host to container
docker cp /path/on/host/file.txt container_name:/path/in/container/

# Copy directory (recursive by default)
docker cp /path/on/host/directory container_name:/path/in/container/

# Copy from container to host
docker cp container_name:/path/in/container/file.txt /path/on/host/
```

### Mount Shared Directory

Share a directory between host and container:

```bash
docker run -it -p 5900:5900 -v ~/shared:/shared mdmitry1/python313-dev:latest
```

Now `~/shared` on host equals `/shared` in container - changes are immediately visible in both locations.

## Common GUI Applications

```bash
# Terminal emulator
xterm &

# Text editor with good clipboard support
apt-get install -y vim-gtk3
gvim &

# Test X11 display
apt-get install -y x11-apps
xeyes &
xclock &
```

## Troubleshooting

### VNC Connection Refused

Make sure VNC server is running inside the container:
```bash
ps aux | grep x11vnc
```

If not running, start it:
```bash
x11vnc -display :99 -forever -nopw -shared -rfbport 5900
```

### Clipboard Not Working

1. Ensure autocutsel is running in both host and container:
   ```bash
   ps aux | grep autocutsel
   ```

2. Test clipboard manually:
   ```bash
   # In container
   echo "test" | xclip -selection clipboard
   xclip -selection clipboard -o
   ```

3. For xterm, use `Shift+Insert` to paste instead of `Ctrl+V`

4. Consider using gvim instead of xterm for better clipboard support

### GUI App Won't Start

Verify DISPLAY is set correctly:
```bash
echo $DISPLAY  # Should show :99
export DISPLAY=:99
```

Check if Xvfb is running:
```bash
ps aux | grep Xvfb
```

## Network Usage

Uploading/downloading the ~2GB compressed image on ADSL: 
- Upload time: ~2 hours
- Download time: varies by connection

Use `screen` or `tmux` for long-running pushes:
```bash
screen -S docker-push
docker push mdmitry1/python313-dev:latest
# Press Ctrl+A then D to detach
# Reconnect: screen -r docker-push
```

## Advanced Usage

### Running as Non-Root User

```bash
docker run -it \
    --user $(id -u):$(id -g) \
    -p 5900:5900 \
    mdmitry1/python313-dev:latest
```

### Persistent Storage Recommendation

```bash
docker run -it \
    -p 5900:5900 \
    -v ~/persistent-data:/data \
    mdmitry1/python313-dev:latest
```

### Custom Screen Resolution

```bash
# Inside container, stop Xvfb and restart with different resolution
pkill Xvfb
Xvfb :99 -screen 0 1920x1080x24 &
```

## System Requirements

- **Host OS:** Linux, WSL2, or Docker Desktop
- **Docker:** Version 29.2.0
- **Disk Space:** ~8GB for image
- **Memory:** 3GB+ recommended
- **Network:** For VNC access and Docker Hub pulls

## Support

For issues or questions:
- Check Docker Hub: https://hub.docker.com/r/mdmitry1/python313-dev
- Review this README
- Check Docker and VNC logs

## Version History

- **latest** - Initial release with Python 3.13, VNC support, and GUI capabilities
