#!/bin/zsh
# Sign dist/FKAstro-pxm.dylib, copy it into PixInsight's bin/ and launch
# PixInsight with the module installed (-m).
#
#   ./install.sh                     # prompts for the .xssk password
#   XSSK_PASSWORD=... ./install.sh
#
# Requires a local signing identity already registered in PixInsight
# (Edit > Local Signing Identity..., "persistent" checked) with the same
# .xssk file. PixInsight must not be running.
set -e

PI=${PI:-/Applications/PixInsight}
PIEXE=$PI/PixInsight.app/Contents/MacOS/PixInsight
XSSK=${XSSK:-$HOME/PixInsight-keys/local.xssk}
DIR=${0:A:h}
MOD=$DIR/dist/FKAstro-pxm.dylib

[[ -f $MOD ]] || { echo "Missing $MOD - run make first"; exit 1; }
[[ -f $XSSK ]] || { echo "Missing signing keys file $XSSK"; exit 1; }

if pgrep -x PixInsight >/dev/null; then
   echo "PixInsight is running. Quit it normally (Cmd+Q) first, so the local"
   echo "signing identity is saved to settings, then run this script again."
   exit 1
fi

if [[ -z $XSSK_PASSWORD ]]; then
   read -s "XSSK_PASSWORD?Password for $XSSK: "; echo
fi

echo "== Signing module"
"$PIEXE" -n --automation-mode --no-startup-scripts --no-splash \
   --sign-module-file="$MOD" --xssk-file="$XSSK" --xssk-password="$XSSK_PASSWORD" \
   --force-exit 2>&1 | grep -E "signature|rror" || true
[[ ${MOD:r}.xsgn -nt $MOD ]] || { echo "Signing failed"; exit 1; }

echo "== Copying to $PI/bin (sudo)"
sudo cp "$MOD" "${MOD:r}.xsgn" "$PI/bin/"

echo "== Launching PixInsight with the module"
open -a "$PI/PixInsight.app" --args -m="$PI/bin/FKAstro-pxm.dylib"
echo "Done. Look for Process > Preprocessing > FKAstro > FKAstroInference."
echo "Quit PixInsight normally once so the module installation is saved."
