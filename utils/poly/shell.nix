{ pkgs , stdenv }:

pkgs.mkShell {

  packages = [ pkgs.nodejs pkgs.python3 ];

  inputsFrom = [ pkgs.bat ];

  shellHook = ''
    echo "welcome to the shell!"
  '';

  test = "AAAAAA";
  ENVVAR = "testtt";

  LD_LIBRARY_PATH = 
    "${pkgs.lib.makeLibraryPath [pkgs.ncurses]}";
  
  RUST_BACKTRACE = 1;

}
