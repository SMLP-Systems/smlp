## Step 1:

Download [RealVNC®](https://www.realvnc.com/en/connect/download/viewer)

## Step 2:

Install RealVNC

## Step 3: Forward Port 5900 from Windows to WSL2

### Step 3.1 - in WSL2 window

Get your WSL2 IP address from running below command:
```bash
hostname -I
```

### Step 3.2

Open Command Prompt and choose **Run as administrator** option

### Step 3.3 - in Windows Command Prompt Window

Use the **first IP** in the output (e.g., `172.31.26.155`). All the rest should be ignored
Run the following in **powershell**, replacing `<WSL2_IP>` with your IP:

```powershell
netsh interface portproxy add v4tov4 listenport=5900 listenaddress=0.0.0.0 connectport=5900 connectaddress=<WSL2_IP>
```

Allow the port through Windows Firewall:
```powershell
New-NetFirewallRule -DisplayName "WSL2 VNC" -Direction Inbound -Protocol TCP -LocalPort 5900 -Action Allow
```

Verify the proxy is set:
```powershell
netsh interface portproxy show all
```

## Step 4: Connect with VNC 

**Connection should be performed after running** `./start_vnc` **command within Docker container**

1. Launch VNC
   Signing in VNC is optional
2. In VNC connect to: `locahost:5900` 
- Ignore non-secure connection warning

## Updating the Port Proxy After WSL2 Restart

WSL2's IP address may change after restart. In this case, **Step 3** should be repeated after the reboot
