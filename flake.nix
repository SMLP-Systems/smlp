# flake.nix
{
  description = "DevShell for a Meson + Boost project";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in pkgs.mkShell {
      nativeBuildInputs = [
        pkgs.pkg-config
      pkgs.meson
        pkgs.ninja
        pkgs.gmp
      pkgs.cmake          # optional – some Boost parts use CMake detection
        pkgs.python312        # optional – if you have a Python helper script
      ];

      buildInputs = [
        pkgs.boost
      ];

      shellHook = ''
        # export BOOST_ROOT=${pkgs.python312Packages.boost}
        # export BOOST_INCLUDEDIR=${pkgs.python312Packages.boost}/include
        # export BOOST_LIBRARYDIR=${pkgs.python312Packages.boost}/lib
        echo "✅  DevShell ready – Meson ${pkgs.meson.version}, Boost ${pkgs.boost.dev.version}"
      '';
    };
  };
}
