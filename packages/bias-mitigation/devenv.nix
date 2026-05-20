{ pkgs, ... }:
{
  imports = [ ./modules/python.nix ];
  env = {
    PREK_ALLOW_NO_CONFIG = "1";
    LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [ pkgs.zlib pkgs.stdenv.cc.cc.lib ]}";
  };
  devcontainer.enable = true;
  dotenv.enable = true;
  packages = with pkgs; [
    gnumake
    cmake
    extra-cmake-modules
    zlib
    uv
    ruff
    black
    isort
    zip
    ninja
    zeromq
    litellm
    sqlite
    gh
    openssh
    pre-commit
    podman
    podman-compose
    texliveFull
  ];

  git-hooks.addGcRoot = false;
  git-hooks.hooks = {
    black.enable = true;
    ruff.enable = false;
    mypy.enable = false;
  };

  dotenv.disableHint = true;
  cachix.enable = false;
}
