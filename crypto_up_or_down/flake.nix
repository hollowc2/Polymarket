{
  description = "Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            uv
            prek
            ruff
            ty
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            # Auto-create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi

            # Activate venv for Python deps
            source .venv/bin/activate

            # Prioritize Nix tools over venv binaries
            export PATH="${pkgs.lib.makeBinPath (with pkgs; [ ruff ty prek ])}:$PATH"

            echo "Python $(python --version) | uv $(uv --version) | ruff $(ruff version)"

            # Keep workspace dependencies in sync
            uv sync --all-packages >/dev/null 2>&1 || true

            # Install pre-commit hooks via prek
            prek install >/dev/null 2>&1 || true
          '';
        };
      });
}
