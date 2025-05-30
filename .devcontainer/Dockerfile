ARG VARIANT=debian-12
FROM mcr.microsoft.com/devcontainers/base:${VARIANT}
USER root

# Install APT packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install tar curl zip unzip bash-completion build-essential ripgrep htop \
    ninja-build ccache g++ mold gdb clang-format clang-tidy \
    rpm pkg-config cmake git smpq gettext libsdl2-dev libsdl2-image-dev libsodium-dev \
    libpng-dev libbz2-dev libfmt-dev libgtest-dev libgmock-dev libbenchmark-dev zsh \
    qtbase5-dev qt6-base-dev ristretto \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install devilutionx-graphics-tools
RUN git clone https://github.com/diasurgical/devilutionx-graphics-tools.git /tmp/devilutionx-graphics-tools && \
    cd /tmp/devilutionx-graphics-tools && \
    cmake -S. -Bbuild-rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build-rel -j $(getconf _NPROCESSORS_ONLN) && \
    cmake --install build-rel --component Binaries && \
    rm -rf /tmp/devilutionx-graphics-tools

# Install devilutionx-mpq-tools
RUN git clone https://github.com/diasurgical/devilutionx-mpq-tools.git /tmp/devilutionx-mpq-tools && \
    cd /tmp/devilutionx-mpq-tools && \
    cmake -S. -Bbuild-rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build-rel -j $(getconf _NPROCESSORS_ONLN) && \
    cmake --install build-rel && \
    rm -rf /tmp/devilutionx-mpq-tools

# Install d1-graphics-tool
RUN curl -O -L https://github.com/diasurgical/d1-graphics-tool/releases/latest/download/D1GraphicsTool-Linux-Qt5.deb && \
    dpkg -i D1GraphicsTool-Linux-Qt5.deb && \
    rm D1GraphicsTool-Linux-Qt5.deb

# Download spawn.mpq and fonts.mpq
RUN curl --create-dirs -O -L --output-dir /usr/local/share/diasurgical/devilutionx/ \
    https://github.com/diasurgical/devilutionx-assets/releases/latest/download/spawn.mpq && \
    curl --create-dirs -O -L --output-dir /usr/local/share/diasurgical/devilutionx/ \
    https://github.com/diasurgical/devilutionx-assets/releases/latest/download/fonts.mpq && \
    chown -R vscode: /usr/local/share/diasurgical/

# Desktop environment configuration
COPY fluxbox /home/vscode/.fluxbox/
