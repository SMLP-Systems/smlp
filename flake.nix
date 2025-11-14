{
    description = "Flake for smlp systems";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    };

    outputs = { self, nixpkgs }:
    let
        pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in
    {
        devShells."x86_64-linux".default = pkgs.mkShell {
            buildInputs = with pkgs; [ gmp meson ninja ];
            installPhase = ''
                meson setup -Dkay-prefix=$HOME/kay --prefix $HOME/.local build
                ninja -C build install
                '';
        };
    };
}
