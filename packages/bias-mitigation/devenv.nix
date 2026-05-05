{ pkgs, ... }:
{
  imports = [ ./modules/python.nix ];

  devcontainer.enable = true;
  dotenv.enable = true;
  packages = with pkgs; [
    gnumake
    cmake
    extra-cmake-modules
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
