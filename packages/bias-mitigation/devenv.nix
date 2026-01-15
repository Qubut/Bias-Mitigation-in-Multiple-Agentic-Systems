{ pkgs, ... }:
{
  imports = [ ./modules/python.nix ];

  devcontainer.enable = true;
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.glib
      pkgs.zlib
      pkgs.libglvnd
      pkgs.xorg.libX11
      pkgs.openssl
      pkgs.zeromq
    ];
  };

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
  ];

  git-hooks.hooks = {
    black.enable = false;
    ruff.enable = true;
    mypy.enable = false;
  };

  dotenv.disableHint = true;
  cachix.enable = false;
}
