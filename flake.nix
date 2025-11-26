{
  description = "Shell for smlp";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs  }: {
    devShells.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;

    in pkgs.mkShell {
      nativeBuildInputs = [
        pkgs.pkg-config
        pkgs.ninja
        pkgs.cmake
        pkgs.python311
      ];

      buildInputs = [
        pkgs.python311Packages.boost
        pkgs.gmp
        pkgs.gmpxx
      ];

      shellHook = ''
        export BOOST_ROOT=${pkgs.lib.getDev pkgs.python311Packages.boost}
        export BOOST_INCLUDEDIR=${pkgs.lib.getDev pkgs.python311Packages.boost}/include
        export BOOST_LIBRARYDIR=${pkgs.lib.getDev pkgs.python311Packages.boost}/lib

        echo "Meson ${pkgs.meson.version}, Boost ${pkgs.python311Packages.boost.version}"

        echo "location being ${pkgs.lib.getDev pkgs.python311Packages.boost}"

        rm -rf build
        meson setup -Dkay-prefix=$HOME/kay --prefix $VIRTUAL_ENV -Dboost-prefix=${pkgs.lib.getDev pkgs.python311Packages.boost} build
        ninja -C build install
      '';
    };
  };
}
