# PUEO Star Camera Server Installation Procedure
_Compiled by **Milan Štubljar** of [stubljar.com](https://stubljar.com) on 2025-01-08 v1.0_

# Configuration

| Hostname | enp1s0 MAC Address   | enp1s0 IP Address  | enp2s0 MAC Address   | enp3s0 MAC Address   | Status       |
|----------|----------------------|--------------------|----------------------|----------------------|--------------|
| erin-01  | `00:04:bf:c0:17:7a`  | 192.168.100.121    | `00:04:bf:c0:9b:8a`  | `00:04:bf:cc:01:5a`  | DHCP/Static  |
| erin-02  | `00:04:bf:c0:ab:b9`  | 192.168.100.55     | `00:04:bf:c0:9b:bc`  | `00:04:bf:cc:8c:f7`  | DHCP/Static  |
| erin-03  | `00:04:bf:c0:a0:d4`  | 192.168.100.61     | `00:04:bf:c0:9b:84`  | `00:04:bf:93:37:84`  | DHCP/Static  |

# Installation Procedure

```
## Install Essentials (Optional)
```bash

# Update/Upgrade All 
sudo apt update
sudo apt upgrade

# Install PyCharm Classic
sudo apt update
sudo apt install snapd 

# Install PyCharm Community
sudo snap install pycharm-community --classic 

# Note: p7zip-full: Installs the main 7zip package.
# Note: p7zip-rar: Adds support for extracting RAR archives.
sudo apt install p7zip-full p7zip-rar

# Install the net-tools package
sudo apt install net-tools
```

- To get the system IP run the next command:
```bash
ifconfig | grep inet
# Example output: 
--->    inet 172.20.10.3  netmask 255.255.255.240  broadcast 172.20.10.15
        inet6 2600:381:9b0f:4ce5:170:313c:38f1:71d4  prefixlen 64  scopeid 0x0<global>
        inet6 2600:381:9b0f:4ce5:d762:8102:722e:2738  prefixlen 64  scopeid 0x0<global>
        inet6 fe80::6019:246d:e341:383  prefixlen 64  scopeid 0x20<link>
--->    inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>

Note: ---> The IPs - localhost, and machine IP for localnetwork remote access.
```

The above IP needs to be updated and set in the conf/config.ini under the STAR_COMM_BRIDGE:
This change is required for both configs on Server and GUI.

```ini
[STAR_COMM_BRIDGE]
# IP address of the server for communication
# Update actual ip from the server if you want local network remote connection.
# In the above example the server_ip should be changed to 172.20.10.3
# server_ip = 127.0.0.1
server_ip = 172.20.10.3
```


## Download and Extract PCC and PCC-GUI Archives (Optional)
```bash
# Navigate to Projects folder
cd /home/pueo-star-tracker2/Projects

# Download archives
wget https://demo.stubljar.com/tmp/Windell/pcc.7z --no-check-certificate
wget https://demo.stubljar.com/tmp/Windell/pcc-gui.7z --no-check-certificate

# Extract PCC
7za x "pcc.7z" -o"pcc"

# Extract PCC-GUI
7za x "pcc-gui.7z" -o"pcc-gui"
```

## Upgrade pcc on sbc: Sync from Github (Upgrade)

For syncing the latest version of the codebase from GitHub to local linux system (Pueo SBCs) use the ```sync_pcc.sh``` (not part of Github)

**Note:** Prior using the sync_pcc.sh make sure the Github user and token are correct, updated and valid:

Edit ```sync_pcc.sh``` and doublecheck on ```GITHUB_USER``` and ```GITHUB_TOKEN```, example:

```bash
GITHUB_USER="milc"       # GitHub username
GITHUB_TOKEN="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"     # PAT from Step 1
```

To run sync simply run: ```./sync_pcc.sh```

Troubleshooting: In the following example, the token has expired and an error is displayed like so:
```bash
fimilc@zaphod:~/Projects/pcc$ ./sync_pcc.sh 
DEBUG: { "message": "Bad credentials", "documentation_url": "https://docs.github.com/rest", "status": "401" }
[ERROR]  pueo_star_camera_operation_code.py
GitHub API Error: Bad credentials
Status: 401
Documentation: https://docs.github.com/rest
Cannot continue. API Error.
Sync failed due to API error!
```
### How to update or generate new token for the script on github
To update or generate a new Personal Access Token (PAT) for your script on GitHub, follow these steps. I'll also include some best practices and troubleshooting tips based on the search results.

### 🔑 1. **Generate a New Personal Access Token**
   - **Log in to GitHub**: Go to [GitHub.com](https://github.com/) and sign in to your account.
   - **Access Settings**: Click on your profile picture in the top-right corner and select **Settings**.
   - **Developer Settings**: In the left sidebar, click on **Developer settings**.
   - **Personal Access Tokens**: Select **Personal access tokens** > **Tokens (classic)** or **Fine-grained tokens** depending on your needs.
   - **Generate New Token**: Click on **Generate new token** (classic) or **Generate new token** (fine-grained).
   - **Configure Token**:
     - **Token description**: Add a descriptive name (e.g., "Script Authentication").
     - **Expiration**: Set an expiration date. For security, avoid using "no expiration".
     - **Scopes/Permissions**: Select the necessary permissions. For script access to repositories, typically **repo** permissions are sufficient. Fine-grained tokens allow more granular control.
   - **Generate**: Click **Generate token**. **⚠️ Important**: Copy the token immediately and store it securely, as you won't be able to see it again.

## LVM + RAID 0 (Best of Both Worlds)
```bash
# TODO: Add instructions for creating raid.

```

## Mount sdcard (if not already)
```
# List available disks
lsblk -f
lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINT

# Check filesystem
sudo blkid /dev/mmcblk0p1
>>> /dev/mmcblk0p1: UUID="5BAB-1DFD" BLOCK_SIZE="512" TYPE="exfat"

# Create Mount Point
sudo mkdir -p /mnt/ssd1
sudo mkdir -p /mnt/ssd2
sudo mkdir -p /mnt/sdcard1

# SD card - NTFS/exFAT - install drivers
# NO LONGER WORKS: sudo apt install ntfs-3g exfat-fuse exfat-utils
# sudo apt install ntfs-3g exfat-fuse exfatprogs
sudo apt install ntfs-3g exfat-fuse exfatprogs
sudo apt policy exfatprogs

# Edit and add to fstab
sudo nano /etc/fstab

# For NTFS/exFAT: 
# NOTE USE the UUID from the blkid command above
# UUID=5BAB-1DFD /mnt/sdcard1 exfat defaults,uid=1000,gid=1000 0 2


# UUID=xxxx-xxxx  /mnt/ssd1  ext4  defaults  0  2
# Example:
UUID=05b2602a-464b-4d70-8374-b5b56d27230f /mnt/ssd1 ext4 defaults 0 0
UUID=56292189-c2bb-4623-b739-542ce5086b26 /mnt/ssd2 ext4 defaults 0 0
UUID=5BAB-1DFD  /mnt/sdcard1  exfat  defaults,uid=1001,gid=1001,fmask=0022,dmask=0022  0  2

# pst01 ssd1
# UUID=a96416fc-cc33-4840-80d5-1e1432102d07 /mnt/ssd1 ext4 defaults 0 0


# First, unmount the SD card from its current location: - IF AUTOMOUNTED to MEDIA
# Optional if required:
sudo umount /dev/mmcblk0p1
sudo umount /mnt/sdcard1
systemctl daemon-reload

# Test the mount
sudo mount -a

# Verify Mount
ls /mnt/sdcard1

```
## Create links for ssd/sd folders

```bash
# Note the ssd1 is in final version raid

cd /home/pst/Projects/pcc
# Remove project placeholder folders (folder should be empty)
# WARNING!!! DO NOT run this if the mount has already been created.
rm ssd_path
rm sd_card_path
sudo mkdir /mnt/raid1/pueo_images_final
sudo mkdir /mnt/raid1/pueo_images_raw
sudo mkdir /mnt/raid1/pueo_autogain
sudo mkdir /mnt/sdcard/pueo_images_ds

sudo chown pst:pst /mnt/raid1/pueo_images_final
sudo chown pst:pst /mnt/raid1/pueo_images_raw
sudo chown pst:pst /mnt/raid1/pueo_autogain
sudo chown pst:pst /mnt/sdcard/pueo_images_ds

# mkdir /mnt/sd/pueo_images
# Create symbolic link from ssd_path to actual ssd folder
ln -s /mnt/ssd1/pueo_images_final output
ln -s /mnt/ssd1/pueo_images_raw ssd_path
ln -s /mnt/ssd1/pueo_autogain autogain

# Create symbolic link from ssd_path to actual sd folder
ln -s /mnt/sdcard1/pueo_images_ds sd_card_path
```

## Install PCC and PCC-GUI
After the .venv is created. Still in PyCharm, open PyCharm terminal and:
```bash
#  Manually creating .venv
sudo apt install python3-pip
sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

```

1. Install tkinter (required only for PCC-GUI)

```bash
sudo apt install python3-tk 
```
2. Install within virtual environment:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install Cedar-Detect prerequisites:
```bash
. .venv/bin/activate
# pip3 install grpcio
# pip3 install grpcio-tools

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install cargo
sudo apt-get install rustup
sudo apt-get install protobuf-compiler
rustup default stable
```

3.1 Run cedar_detect
```bash
cd ~/Projects/pcc/lib/cedar_detect/python

# Do not run this!!! Files are slightly modified to accommodated for our own location of cedar-detect
# ../../../.venv/bin/python -m grpc_tools.protoc -I../src/proto --python_out=. --pyi_out=. --grpc_python_out=. ../src/proto/cedar_detect.proto

cargo run --release --bin cedar-detect-server

# Looks like this:
pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc/lib/cedar_detect/python$ cargo run --release --bin cedar-detect-server 
   Compiling imageproc v0.25.0
   Compiling cedar_detect v0.7.0 (/home/pueo-star-tracker2/Projects/pcc/lib/cedar_detect)
   Compiling prctl v1.0.0
   Compiling tonic-web v0.11.0
   Compiling prost-types v0.12.6
    Finished `release` profile [optimized] target(s) in 2m 30s
     Running `/home/pueo-star-tracker2/Projects/pcc/lib/cedar_detect/target/release/cedar-detect-server`
[2025-03-18T09:38:55Z INFO  cedar_detect_server] CedarDetectServer listening on 0.0.0.0:50051
```

3. Install astrometry.net

**Note:** Requires INTERNET access.

```bash
sudo apt update
sudo apt upgrade
sudo apt install astrometry.net 

```


## Configure USB for Telemetry/Focuser
Next steps are required to allow the regular user to connect to USB.

Run the code below and then logout/login.
```bash
# sudo usermod -a -G dialout pueo-star-tracker2
sudo usermod -a -G dialout pst
```
## Install ASI Camera SDK
```
sudo apt install bzip2
```

## Install; the ASI SDK Library:
Find the installation instructions.

NEXT TRY WITHOUT THIS!!!
- Only do the ASIStudio from camera folder, running run.
- Extract ASI Camera SDK to get out the .so and asi.rules

```bash

# Add the PPA to Your System:
# Open a terminal and execute the following command to add the PPA:
sudo bash -c 'echo "deb [trusted=yes] https://apt.fury.io/jgottula/ /" > /etc/apt/sources.list.d/jgottula.list'

# Update Package Lists and Install the Library:
# After adding the PPA, update your package lists and install the libasicamera2 package:
# SKIP NEXT 3 STEPS!!!
sudo apt update
sudo apt install libasicamera2
sudo apt remove libasicamera2

# Verify the Installation:
# Check for the Library File:
# Ensure that the library file libASICamera2.so is present in /usr/local/lib. You can verify this by running:
ls /usr/local/lib | grep libASICamera2.so

# Update the Library Cache:
# After installation, update the library cache to recognize the new library:
sudo ldconfig

# 
cd ~/Projects/camera
bunzip2 ASI_linux_mac_SDK_V1.37.tar.bz2 
tar -xvf ASI_linux_mac_SDK_V1.37.tar

# Install ASIStudio
cd ~/Projects/camera
chmod +x ASIStudio_V1.14.run
./ASIStudio_V1.14.run

# Copy the lib to ASIStudio
cd /home/pst/Projects/camera/ASI_linux_mac_SDK_V1.37/lib/x64
cp libASICamera2.so ~/ASIStudio/

# Configure Udev Rules:

# To allow non-root users to access the camera, you need to set up appropriate udev rules:

# Download and Install Udev Rules:
# If the udev rules file (asi.rules) was included in your installation, copy it to the udev rules directory:
# The asi.rules can be fund in lib folder of the SDK.
cd ~/Projects/camera/ASI_linux_mac_SDK_V1.37/lib
sudo cp asi.rules /etc/udev/rules.d/

# Reload Udev Rules:
# After copying the rules file, reload the udev rules:

sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Configure SERIAL Ports (not USB)
```bash
# Identify Active Serial Ports:
# Use the setserial command to check the configuration and status of each serial port:

sudo apt-get install setserial  # Install setserial if it's not already installed
sudo setserial -g /dev/ttyS[0-3]
```

## Install VersaLogic API Package
Download package e.g.: VersaAPI_Linux_64b_v1.8.4.tar e.g. to folder ~/Projects/install
Follow the instructions on the API docs, page 25, file: MEPU-4012_PRM.pdf
```bash
cd ~/Projects/install
tar -xvf VersaAPI_Linux_64b_v1.8.4.tar

uname -r 
# Yields> 6.8.0-52-generic
# Use output to install linux-headers e.g.
# sudo apt install linux-headers-6.8.0-52-generic
#                              ----------------

# Install 
chmod +x vl_install.sh
sudo ./vl_install.sh EPU-4012
 
```


## Install Required Tools & Libraries for Telemetry (CPU load, temp, disk ...)
- psutil – CPU/RAM/Disk usage

```bash
sudo apt install lm-sensors
sudo sensors-detect  # (Follow prompts to detect hardware)
sudo apt install nvme-cli
sudo apt install smartmontools
```

## Install AUTORUN at startup

```txt
Given the ubuntu it is required to 
  a. Autologin a user: pueo-star-tracker2
  b. Start a couple of scripts after each boot including a script that requires sudo.

1. Install the VersaLogic API Package (the vldriver disapears after reboot), needs to run as sudo
Command 1: 
     cd ~/Projects/install
     sudo ./vl_install.sh EPU-4012

# 2. Start custom star detecting cedar-detect rust server as regular user
# Command 2:
    cd ~/Projects/pcc/lib/cedar-detect/python
    cargo run --release --bin cedar-detect-server & 

3. Start pcc as regular user
Command 3:
    cd ~/Projects/pcc
    ./.venv/scripts/python pueo_star_camera_operation_code.py &

Devise steps required to get the aboce automated either as single script or edits of existing startup scripts. As questions if further information is required.

===>>>
```

### RENAMING HOSTNAMES
```bash 
# Using hostnamectl (Recommended)
sudo hostnamectl set-hostname pst01
sudo hostnamectl set-hostname pst02
sudo hostnamectl set-hostname pst03
```
### CREATE NEW USER PST
```bash
# Create the user:
sudo adduser -m pst

# Set a password:
sudo passwd pst

# Add to the sudo group:
sudo usermod -aG sudo pst

# MOVE HOME to
# 1. Create the new home directory on the SSD 
sudo mkdir -p /mnt/ssd1/home/pst
sudo chown pst:pst /mnt/ssd1/home/pst  # Set correct ownership

# 2. Copy all files from the old home to the new location
sudo rsync -avz /home/pst/ /mnt/ssd1/home/pst/

# 3. Rename the old home folder (as a backup)
sudo mv /home/pst /home/pst.backup

# 4. Create a symlink from /home/pst to the new location
sudo ln -s /mnt/ssd1/home/pst /home/pst
sudo chown pst:pst /home/pst  # Ensure ownership is correct

# 5. Verify the symlink
ls -l /home/pst
# =>>> lrwxrwxrwx 1 root root 17 Apr 20 12:34 /home/pst -> /mnt/ssd1/home/pst

# 6. Test the new setup
sudo rm -rf /home/pst.backup

# Switch to user
su - pst
```

## Automate AUTOSTART of PUEO STAR TRACKER SERVER
To automate the autologin and startup scripts on Ubuntu, we'll need to perform several steps. Here's a comprehensive solution:

### A. Set Up Autologin for `pst`

1. **For Ubuntu with GNOME (or GDM3)**:  
   Edit the GDM3 configuration file:
   ```bash
   sudo nano /etc/gdm3/custom.conf
   ```
   Add/modify these lines under `[daemon]`:
   ```
   [daemon]
   AutomaticLoginEnable=true
   AutomaticLogin=pst
   ```

### B. Automate Startup Scripts

Since one of the commands requires `sudo`, we'll use a combination of `systemd` (for the `sudo` command) and `autostart` (for user-level commands).

#### 1. **For the `sudo` command (vl_install.sh)**  
Create a systemd service to run at boot:

```bash
sudo nano /etc/systemd/system/vl_install.service
```

Add the following content:
```ini
[Unit]
Description=VersaLogic API Installation
After=network.target

[Service]
Type=oneshot
User=pst
WorkingDirectory=/home/pst/Projects/install
ExecStart=/usr/bin/sudo ./vl_install.sh EPU-4012

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl enable vl_install.service
sudo systemctl enable --now vl_install.service

```

#### 2. **For User-Level Commands (cedar-detect-server & pueo_star_camera_operation_code.py)**  
Use the user's autostart directory to run these at login.

1. Create a shell script to run the commands:
   ```bash
   mkdir -p ~/scripts
   nano ~/scripts/startup_commands.sh
   ```
   Add:
   ```bash
   #!/bin/bash
   cd ~/Projects/pcc/lib/cedar-detect/python
   cargo run --release --bin cedar-detect-server &

   cd ~/Projects/pcc
   ./.venv/bin/python pueo_star_camera_operation_code.py &
   ```
   Make it executable:
   ```bash
   chmod +x ~/scripts/startup_commands.sh
   ```

2. Add a desktop entry to autostart:
   ```bash
   mkdir -p ~/.config/autostart
   nano ~/.config/autostart/startup_commands.desktop
   ```
   Add:
   ```ini
   [Desktop Entry]
   Type=Application
   Name=Startup Commands
   Exec=/home/pueo-star-tracker2/scripts/startup_commands.sh
   Hidden=false
   X-GNOME-Autostart-enabled=true
   ```

### C. Verify Permissions & Dependencies (Do VIA Sudo Rule next section)

- Ensure `pst` has passwordless sudo for `vl_install.sh`:
  ```bash
  sudo visudo
  ```
  Add:
  ```
  # pueo-star-tracker2 ALL=(ALL) NOPASSWD: /home/pueo-star-tracker2/Projects/install/vl_install.sh
  pst ALL=(ALL) NOPASSWD: /home/pst/Projects/install/vl_install.sh
  ```
- Make sure `cargo` and Python dependencies are installed.

It's a good practice to use `/etc/sudoers.d/` for adding custom sudo rules rather than modifying the main `/etc/sudoers` file directly. Here's how you can do it:

### Steps to Add the Sudo Rule via `/etc/sudoers.d/`:

1. **Create a new sudoers file** (e.g., `pst`) in `/etc/sudoers.d/`:
   ```bash
   sudo visudo -f /etc/sudoers.d/pst
   ```
   This will open the file in a safe way (with syntax checking).

2. **Add your rule** to the file:
   ```
   pst ALL=(ALL) NOPASSWD: /home/pst/Projects/install/vl_install.sh
   ```

3. **Save and exit** (`Ctrl+X`, then `Y` if using `nano`, or `:wq` if using `vim`).

4. **Verify the permissions** of the file:
   ```bash
   sudo chmod 440 /etc/sudoers.d/pst
   ```
   (This ensures only `root` can read/write it.)

5. **Test the rule**:
   ```bash
   sudo -u pueo-star-tracker2 sudo -l
   ```
   (This should show that the user can run the script without a password.)

### Why This is Better:
- **Modularity**: Keeps custom rules separate from the main `sudoers` file.
- **Safety**: Easier to manage and debug.
- **Prevents Errors**: `visudo` checks syntax before saving, reducing the risk of breaking sudo.

### D. Reboot & Test
```bash
sudo reboot
```

After reboot:
- Check if the user autologins.
- Verify services:
  ```bash
  systemctl status vl_install.service
  ps aux | grep -E "cedar-detect-server|pueo_star_camera_operation"
  ```

### Handling of console log
- logrotate with Copy-Truncate (Best for Production)

Create a config file (sudo nano /etc/logrotate.d/pueo_console):
```bash
sudo nano /etc/logrotate.d/pueo_console
```

```conf
/home/pst/Projects/pcc/logs/pueo_console.log {
    hourly
    size 128M
    rotate 10
    copytruncate
    compress
    missingok
    notifempty
}
```
- 
- Add an Hourly Cron Job (Recommended)
```
sudo crontab -e
# Add:
0 * * * * /usr/sbin/logrotate -f /etc/logrotate.d/pueo_console

```

- Fix permissions (restrict to owner only):
```bash
ls -ld /home/pst/Projects/pcc/logs/
sudo chmod 750 /home/pst/Projects/pcc/logs/
sudo chown pst:pst /home/pst/Projects/pcc/logs/  # Ensure correct ownership
```
Test it manually:

```
# Force run:
sudo logrotate -vf /etc/logrotate.d/pueo_console

# Dry run:
df -

systemctl list-timers --all | grep logrotate

```

## How to Run the SERVER from Terminal
Open terminal and navigate to the SBC pcc server folder. 
```bash
cd /home/pueo-star-tracker2/Projects/pcc

# Only once after the installation
chmod +x run.sh

# Run server:
./run.sh > ./logs/server-console.log &

```

## Single Board Computer (SBC) Configuration
```bash

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ cat /etc/debian_version 
trixie/sid


(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.1 LTS
Release:        24.04
Codename:       noble

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ df -h 
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           777M  1.9M  775M   1% /run
/dev/mmcblk0p2   28G   14G   13G  52% /
tmpfs           3.8G  1.3M  3.8G   1% /dev/shm
tmpfs           5.0M  8.0K  5.0M   1% /run/lock
efivarfs        558K   61K  493K  11% /sys/firmware/efi/efivars
/dev/sda        3.6T  380M  3.4T   1% /mnt/ssd1
/dev/sdb        3.6T   32K  3.4T   1% /mnt/ssd2
/dev/mmcblk0p1  1.1G  6.2M  1.1G   1% /boot/efi
tmpfs           777M  128K  777M   1% /run/user/1000
/dev/mmcblk1p1  470G  768K  470G   1% /media/pueo-star-tracker2/5BAB-1DFD

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ lscpu
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   4
  On-line CPU(s) list:    0-3
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Atom(TM) Processor E3950 @ 1.60GHz
    CPU family:           6
    Model:                92
    Thread(s) per core:   1
    Core(s) per socket:   4
    Socket(s):            1
    Stepping:             10
    CPU(s) scaling MHz:   100%
    CPU max MHz:          2000.0000
    CPU min MHz:          800.0000
    BogoMIPS:             3187.20

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ nproc
4

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ free -h
               total        used        free      shared  buff/cache   available
Mem:           7.6Gi       3.1Gi       2.0Gi       400Mi       3.1Gi       4.4Gi
Swap:          4.0Gi          0B       4.0Gi

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ lsusb
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 05e3:0610 Genesys Logic, Inc. Hub
Bus 001 Device 003: ID 03c3:294a ZWO ASI294MM
Bus 001 Device 004: ID 413c:3200 Dell Computer Corp. Mouse
Bus 001 Device 005: ID 3938:1032 MOSART Semi. 2.4G RF Keyboard & Mouse
Bus 001 Device 006: ID 413c:2107 Dell Computer Corp. KB212-B Quiet Key Keyboard

# Arduino
Bus 001 Device 007: ID 1a86:7523 QinHeng Electronics CH340 serial converter

Bus 001 Device 012: ID 05ac:12a8 Apple, Inc. iPhone 5/5C/5S/6/SE/7/8/X/XR
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub

(.venv) pueo-star-tracker2@pueo-star-tracker2-EPU-4011-4012:~/Projects/pcc$ lsusb
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 05e3:0610 Genesys Logic, Inc. Hub
Bus 001 Device 004: ID 413c:3200 Dell Computer Corp. Mouse
Bus 001 Device 005: ID 3938:1032 MOSART Semi. 2.4G RF Keyboard & Mouse
Bus 001 Device 006: ID 413c:2107 Dell Computer Corp. KB212-B Quiet Key Keyboard
Bus 001 Device 012: ID 05ac:12a8 Apple, Inc. iPhone 5/5C/5S/6/SE/7/8/X/XR
Bus 001 Device 013: ID 03c3:294a ZWO ASI294MM
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub


```
## Setting permissions for USB/Serial
```
# Should be Ardoino:
sudo dmesg | grep -i "usb\|tty"
# >>> usb 1-6: ch341-uart converter now attached to ttyUSB0

# sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB0

# For Focuser
sudo chmod 666 /dev/ttyS0
```
## Setting Serial for Console/Getty Remote Backup Access
### Serial Port Redirection

```bash
sudo nano /etc/default/grub

GRUB_CMDLINE_LINUX_DEFAULT=""
GRUB_CMDLINE_LINUX="console=tty0 console=ttyS0,115200n8"
GRUB_TERMINAL="console serial"
GRUB_SERIAL_COMMAND="serial --speed=115200 --unit=0 --word=8 --parity=no --stop=1"

sudo update-grub
```
# Getty Setup 
```bash
# Install getty
sudo apt-get update
sudo apt-get install util-linux

# Configure a service for getty
# Create or Modify the Getty Service
# Use systemd’s built-in serial-getty@.service (Recommended)
sudo systemctl enable serial-getty@ttyS0.service
sudo systemctl start serial-getty@ttyS0.service
sudo systemctl edit serial-getty@ttyS0

# Add this to the file:
[Service]
ExecStart=
ExecStart=-/sbin/agetty -o "-p -- \\u" -s 115200 %I $TERM
Restart=always
RestartSec=5s

# Save and
sudo systemctl daemon-reload
# Stop and Start or Restart Service 
sudo systemctl stop serial-getty@ttyS0
sudo systemctl start serial-getty@ttyS0
sudo systemctl restart serial-getty@ttyS0

# Check status
sudo systemctl status serial-getty@ttyS0.service

# Test - Connection - On Flight Computer, assuming the ttyS0 is also configured to connect to Erin
sudo apt-get install minicom

sudo minicom -D /dev/ttyS0 -b 115200

#
# Troubleshooting
#
# If getty isn't starting, check logs:
journalctl -u serial-getty@ttyS0.service

# Verify the serial port is available:
ls -l /dev/ttyS0

# Expected output: crw-rw---- 1 root dialout 4, 64 ...  
# If not, fix permissions:
sudo usermod -a -G dialout $USER
sudo chmod 660 /dev/ttyS0

# Check dmesg for serial port detection:
dmesg | grep ttyS
```

## Setting up Chamber Test Mode
1. Key configuration settings (conf/config.ini):

```ini
# Camera ASI SDK Library
env_filename = /home/pueo-star-tracker2/ASIStudio/libASICamera2.so
# Focuser Port connected to hardware serial port
focuser_port = /dev/ttyS1
# Arduino Telemtry Temperature Sensors connected to USB2
telemetry_port = /dev/ttyUSB0
server_ip = 172.20.10.3
```
 
2. Enable Chamber and Autonomous Mode at Startup
```ini
# flight_mode: str: preflight, flight
#   preflight: normal operation but images not saved (RAW, FINAL, ...)
#   flight: normal operation
flight_mode = 'flight'

# Enable autonomous mode at startup
run_autonomous = True
# run_telemetry, set to True to enable collecting telemetry data at all times to logs/telemetry.log
run_telemetry = True
# run_chamber, set to True to enable normal camera capture but serve images from test_images folder
# a mode used for testing in a dark chamber, to generate heat by normal operation yet getting images for solving
run_chamber = True
```

3. Run Server
```bash
# Use Pycharm or command line/terminal
cd /home/pueo-star-tracker2/Projects/pcc

# Run server:
./run.sh > ./logs/server-console.log &
```

The **telemetry** is saved to the logs/telemetry.log of the server installation folder.

## Troubleshooting 
### Exposure Time and Image Capture
Turns out that the camera.capture() on SBC1 fails when the exposure time is set to less than 1000000 (1s). The previous settings e.g. 30x1000, worked on dev. laptop.

### Power issues
Make sure sufficient power is available for all USB/serial devices connected.


### Folder Size Monitor
1m/10m/30m/60m deltas, MB reporting, and logging to `/home/pst/Projects/pcc/logs/pcc_folder_stats.log`):

### **Script: `folder_monitor.sh` (Symlink-Aware)**
Script location: ```/home/pst/Projects/pcc/logs/folder_monitor.sh```

---

### **Features**
1. **Symlink Resolution**:
   - Added `resolve_symlink()` function to resolve paths like `ssd_path -> /mnt/ssd1/pueo_images_raw/`.
   - Uses `readlink -f` to get absolute target paths.

2. **`find -L` Flag**:
   - The `-L` option makes `find` follow symlinks when scanning directories.

3. **Error Handling**:
   - Skips directories/symlinks if their resolved target doesn’t exist (`NOT_FOUND` in logs).
   - Suppresses `find` errors (`2>/dev/null`) for cleaner operation.

---

### **Example Log Output**
```
2024-04-20 16:30:00,output:Files=50(Δ1m=+2|Δ10m=+10|Δ30m=+25|Δ60m=+50),SizeMB=100.50(Δ1m=+1.20|Δ10m=+5.50|Δ30m=+15.30|Δ60m=+30.75),ssd_path:Files=200(Δ1m=+5|Δ10m=+20|Δ30m=+60|Δ60m=+120),SizeMB=500.25(Δ1m=+2.50|Δ10m=+10.75|Δ30m=+32.30|Δ60m=+65.40),TOTAL:Files=250,SizeMB=600.75
```

---

### **How to Use**
1. **Save and Run**:
   ```bash
   chmod +x folder_monitor.sh
   nohup ./folder_monitor.sh > /dev/null 2>&1 &
   ```
2. **Stop**:
   ```bash
   pkill -f folder_monitor.sh
   ```

---

### **Notes**
- **Performance**: Following symlinks adds minimal overhead.
- **Permissions**: Ensure the script has read access to symlink targets.
- **Log Rotation**: Add logrotate rules for `pcc_folder_stats.log` if long-term logging is needed.


#  Further Monitoring

### **`btop` (Next-gen System Monitor)**
**Description**: A modern, feature-rich replacement for `top` and `htop` with advanced visuals and real-time metrics.  

#### **Installation**  
**Ubuntu/Debian (apt)**:
```bash
sudo apt update && sudo apt install btop
```

#### **Key Features**  
✅ **Multi-pane UI**:  
   - CPU, memory, disks, network, and processes in one view.  
   - Color-coded and mouse-support.  

✅ **Detailed Metrics**:  
   - Per-core CPU usage + frequency.  
   - GPU stats (if supported).  
   - Disk I/O and network bandwidth graphs.  

✅ **Interactive Controls**:  
   - Sort processes by CPU/RAM (`P`/`M`).  
   - Tree view (`T`), kill processes (`k`).  
   - Customizable themes (`F2`).  

#### **Usage**  
1. Start:  
   ```bash
   btop
   ```
2. **Shortcuts**:  
   - `q`: Quit  
   - `h`: Help menu  
   - `+`/`-`: Adjust update speed  

#### **Why Choose `btop`?**  
- **More intuitive** than `top` with a modern UI.  
- **Lightweight** despite rich features.  
- **Extensible**: Supports plugins (e.g., `btop-plugins`).  

![btop screenshot](https://github.com/aristocratos/btop/raw/main/Imgs/main.png) *(Example UI)*  

#### **Uninstall**  
```bash
sudo apt remove btop  # Ubuntu/Debian
# OR
sudo /usr/local/share/btop/uninstall.sh  # Manual install
```

#### Cleanup - Remove files
```bash
# Goto designated folder
# 
cd ~/Projects/pcc/log
cd ~/Projects/pcc/autogain
cd ~/Projects/pcc/output
cd ~/Projects/pcc/sd_card_path
cd ~/Projects/pcc/sdd_path

# ...
# Delete all files in current folder (when rm -rf fails with argument to long)
find . -maxdepth 1 -type f -delete

# Delete all files and folders in current folder
find . -mindepth 1 -delete
```

#### Checks
```bash
# Focuser properly initialised - using debug log parsing
cd ~/Projects/pcc/logs
cat debug-server.log | grep Focuser
# Observe lines "Focuser Initialized Successfully" or other focuser commands should have 'OK' inline...
# Example:
.. INFO Focuser PA (Print Aperture Position): ['pa', 'OK', '0,f14', ''] pos: 0 f_val: f14
# Focuser properly initialised - using cli command
./pc.sh get_aperture_position

# The correct response gives a value like:
"aperture_postion": 0
"aperture_f_val": "f14"

# Camera Initialized:
pst@erin-02:~/Projects/pcc/logs$ cat debug-server.log |grep ZWO
2025-06-05 23:17:44.529 9718   318:        initialize_camera   DEBUG Found one camera: ZWO ASI294MM

# Telemetry recording
pst@erin-02:~/Projects/pcc/logs$ cat telemetry.log | grep Sensor
2025-06-05 23:17:57.548 9718   105:                 __init__    INFO Sensors initialized.
2025-06-05 23:17:57.559 9718   109:                 __init__    INFO Sensors: package_id_0_temp, core_0_temp, core_1_temp, core_2_temp, core_3_temp, acpitz-acpi-0_temp
2025-06-05 23:18:00.408 9718   323:         get_arduino_data    INFO Arduino: Sensor 1 Address: 28D02D3F0000001C
2025-06-05 23:18:00.440 9718   323:         get_arduino_data    INFO Arduino: Sensor 2 Address: 280A363F0000001B
2025-06-05 23:18:00.483 9718   323:         get_arduino_data    INFO Arduino: Sensor 3 Address: 28C6294100000057
2025-06-05 23:18:00.542 9718   323:         get_arduino_data    INFO Arduino: Sensor 4 Address: 2881313F00000058
2025-06-05 23:18:00.616 9718   323:         get_arduino_data    INFO Arduino: Sensor 5 Address: 28D9D63E000000D3
2025-06-05 23:18:00.704 9718   323:         get_arduino_data    INFO Arduino: Sensor 6 Address: 2857803F000000AC

tail -f telemetry.log
# Look for Telemetry header that is printed every 20 or so seconds (lines). It should include the SX sensors, those are Arduino sensors if arduino is properly initialised and present.

2025-06-05 23:34:15.220 9718   403:                 log_data    INFO Telemetry header: package_id_0_temp, core_0_temp, core_1_temp, core_2_temp, core_3_temp, acpitz-acpi-0_temp, core0_load, core1_load, core2_load, core3_load, S1, S2, S3, S4, S5, S6
2025-06-05 23:33:41.568 9718   404:                 log_data    INFO Telemetry data: 2025-06-05 23:33:40, 42.0 °C, 41.0 °C, 41.0 °C, 42.0 °C, 41.0 °C, 41.0 °C, 33.7 %, 33.3 %, 37.0 %, 38.1 %, 27.12 °C, 33.25 °C, 33.38 °C, 28.00 °C, 25.50 °C, 34.25 °C
2025-06-05 23:33:42.645 9718   404:                 log_data    INFO Telemetry data: 2025-06-05 23:33:41, 42.0 °C, 41.0 °C, 42.0 °C, 42.0 °C, 42.0 °C, 41.0 °C, 40.8 %, 34.7 %, 40.6 %, 35.7 %, 27.19 °C, 33.25 °C, 33.31 °C, 28.00 °C, 25.50 °C, 34.25 °C


```

#### Training and Knowledge Transfer
- Basic usage
  - Starting/Stoping the PUEO Server/GUI
  - Check list:
    - Configure (conf/config.ini)
      - preflight mode: flight_mode = preflight
      - autonomous mode: run_autonomous = True
    - Run server, manually or automatically rebooting the system.
      - Enable flight mode by sending cli command: set_flight_mode flight
  - Checking PUEO is running
    - Logs: logs/
      - Console logs, tail -f pueu_console.log
      - Server logs
      - client
      - ./status.sh lists pids of cedar/pueo server, need to kill pueo, use first of the two pids
        - kill -9 <pid>
  - Sending commands to Pueo Server
    - ./pc.sh <command> [options]
    - For example: 
      - ./pc.sh set_flight_mode flight
      - ./pc.sh get_focus
      - ./pc.sh set_focus 1215
      - ...
    - or use ./pci.sh as interactive shell to the PUEO Server, only sending commands:
      - set_flight_mode flight
      - set_focus 1215
      - get_gain

--- 
