{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShell = pkgs.mkShellNoCC {
        venvDir = ".venv";
        packages = with pkgs; [
          (python313.withPackages (python-pkgs:
            with python-pkgs; [
              pip
              torch
              torchvision
            ]))
          iverilog
          ruff
        ];
      };
    }
  );
}
