#!/bin/sh

: ${BIN:=$HOME/.local/bin/nul}
: ${NUL:=$HOME/nul}
: ${PY:=$NUL/.venv/bin/python}

mkdir -p $(dirname $BIN)
cat <<EOF > $BIN
#!/bin/sh
$PY $NUL/main.py "\$@"
EOF

chmod +x $BIN
