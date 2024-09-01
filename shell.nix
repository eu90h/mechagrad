{ pkgs ? import <nixpkgs> {
  config.allowUnfree = true;
} }:
  let
    overrides = (builtins.fromTOML (builtins.readFile ./rust-toolchain.toml));
in
  pkgs.mkShell rec {
    buildInputs = with pkgs; [
      vscode
      clang
      rustup
      pkg-config
      lapack
      gfortran
      libtorch-bin
    ];

    RUSTC_VERSION = overrides.toolchain.channel;
    LIBTORCH = pkgs.libtorch-bin;
    LIBTORCH_INCLUDE = pkgs.libtorch-bin.dev;

    shellHook = ''
      export PATH=$PATH:''${CARGO_HOME:-~/.cargo}/bin
      export PATH=$PATH:''${RUSTUP_HOME:-~/.rustup}/toolchains/$RUSTC_VERSION-x86_64-unknown-linux-gnu/bin/
    '';
  }
