FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

##
## Prepare bash envrionment
##
RUN cat <<'EOF' >> /root/.bashrc

PS1='[\u@\h \W]# '
TERM=linux
PROMPT_COMMAND='history -a; history -n'
shopt -s histappend

# Jump into the venv on each run
source /root/venv/bin/activate
cd /root/devel/DevilutionX-AI/ai
EOF

##
## Install main packages
##
RUN apt-get update -y && \
    apt install -y build-essential git cmake systemd-coredump \
                   gdb pipenv libsdl2-dev libsodium-dev libbz2-dev \
                   libgtest-dev libbenchmark-dev libgmock-dev \
                   python3-pip wget borgbackup tmux screen vim && \
    rm -rf /var/lib/apt/lists/*

##
## Set up Python venv
##
RUN virtualenv /root/venv
COPY ./ai/requirements.txt /tmp/requirements.txt
RUN /root/venv/bin/pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

##
## Copy DevilutionX-AI and build it
##
COPY . /root/devel/DevilutionX-AI-copy
RUN git clone /root/devel/DevilutionX-AI-copy /root/devel/DevilutionX-AI && \
    [ -d /root/devel/DevilutionX-AI-copy/ai/models ] && \
        cp -r /root/devel/DevilutionX-AI-copy/ai/models /root/devel/DevilutionX-AI/ai/ || true && \
    rm -rf /root/devel/DevilutionX-AI-copy

RUN cmake -S /root/devel/DevilutionX-AI -B /root/devel/DevilutionX-AI/build \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DBUILD_TESTING=OFF \
          -DDEBUG=ON \
          -DUSE_SDL1=OFF \
          -DHAS_KBCTRL=1 \
          -DPREFILL_PLAYER_NAME=ON \
          \
          -DKBCTRL_BUTTON_DPAD_LEFT=SDLK_LEFT \
          -DKBCTRL_BUTTON_DPAD_RIGHT=SDLK_RIGHT \
          -DKBCTRL_BUTTON_DPAD_UP=SDLK_UP \
          -DKBCTRL_BUTTON_DPAD_DOWN=SDLK_DOWN \
          -DKBCTRL_BUTTON_X=SDLK_y \
          -DKBCTRL_BUTTON_Y=SDLK_x \
          -DKBCTRL_BUTTON_B=SDLK_a \
          -DKBCTRL_BUTTON_A=SDLK_b \
          -DKBCTRL_BUTTON_RIGHTSHOULDER=SDLK_RIGHTBRACKET \
          -DKBCTRL_BUTTON_LEFTSHOULDER=SDLK_LEFTBRACKET \
          -DKBCTRL_BUTTON_LEFTSTICK=SDLK_TAB \
          -DKBCTRL_BUTTON_START=SDLK_RETURN \
          -DKBCTRL_BUTTON_BACK=SDLK_LSHIFT

RUN make -C /root/devel/DevilutionX-AI/build -j$(nproc --all)

##
## Download Diablo asset
##
RUN wget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq \
          -P /root/devel/DevilutionX-AI/build


##
## Once `sprout.py` is used for the first time, Borg complains with
## the following: "Warning: Attempting to access a previously unknown
## unencrypted repository!"
##
ENV BORG_UNKNOWN_UNENCRYPTED_REPO_ACCESS_IS_OK=yes

# Start new detached tmux session and run bash
CMD ["/bin/bash", "-c", "tmux new-session -d && exec bash"]
