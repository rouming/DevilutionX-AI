FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# Install packages
RUN apt-get update -y
RUN apt install -y build-essential git cmake systemd-coredump \
                   gdb pipenv libsdl2-dev libsodium-dev libbz2-dev \
                   libgtest-dev libbenchmark-dev libgmock-dev \
                   python3-pip wget tmux screen vim

# Build patched DevilutionX
RUN git clone https://github.com/rouming/DevilutionX-AI.git /root/devel/DevilutionX-AI
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

# Download Diablo asset
RUN wget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq \
          -P /root/devel/DevilutionX-AI/build

# Setup Python venv
RUN virtualenv /root/devel/DevilutionX-AI/ai/venv
RUN /root/devel/DevilutionX-AI/ai/venv/bin/pip \
    install -r /root/devel/DevilutionX-AI/ai/requirements.txt

# Exclude the whole path from prompt, make it short
RUN echo >> /root/.bashrc
RUN echo "PS1='[\u@\h \W]# '" >> /root/.bashrc

# TERM setting
RUN echo "TERM=linux" >> /root/.bashrc

# Bash history setting
RUN echo "PROMPT_COMMAND='history -a; history -n'" >> /root/.bashrc
RUN echo "shopt -s histappend" >> /root/.bashrc

# Jump into the venv on each run
RUN echo \
    >> /root/.bashrc
RUN echo "# Jump into the venv on each run" \
    >> /root/.bashrc
RUN echo "source /root/devel/DevilutionX-AI/ai/venv/bin/activate" \
    >> /root/.bashrc
RUN echo "cd /root/devel/DevilutionX-AI/ai" \
    >> /root/.bashrc

# Start new detached tmux session and run bash
CMD ["/bin/bash", "-c", "tmux new-session -d && exec bash"]
