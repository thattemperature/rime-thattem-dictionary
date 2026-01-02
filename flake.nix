{

  description = "Environment for that temperature's rime data process.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils/main";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:

      let

        pkgs = import nixpkgs { inherit system; };

        python-pkgs =
          ps: with ps; [
          ];
        python-env = pkgs.python3.withPackages python-pkgs;

      in

      {
        devShells.default = pkgs.mkShell {
          name = "rime-thattem-dictionary-shell";

          buildInputs = [
            python-env
          ];

          PYTHONPATH = "${python-env}/${python-env.sitePackages}";

        };
      }
    );

}
