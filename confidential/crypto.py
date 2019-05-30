import argparse

from crypt.decrypt import decrypt_blob
from crypt.encrypt import encrypt_blob

EXTENSION = ".encrypt"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
group = ap.add_mutually_exclusive_group()
group.add_argument(
    "-e", action="store_true", help="for encrypt with public key"
)
group.add_argument(
    "-d", action="store_true", help="for decrypt with private key"
)
ap.add_argument(
    "files", metavar="N", type=str, nargs="+", help="path to files for encrypt"
)
ap.add_argument(
    "-k", "--key", required=True, help="path to public/private key file"
)
args = vars(ap.parse_args())

f_key = open(args["key"], "rb")
key = f_key.read()
f_key.close()

for f in args["files"]:
    f_in = open(f, "rb")
    if args["e"]:
        print(f"[INFO] Encrypting: {f}")
        output = encrypt_blob(f_in.read(), key)
        n_out = f + EXTENSION
    elif args["d"]:
        print(f"[INFO] Decrypting: {f}")
        output = decrypt_blob(f_in.read(), key)
        n_out = f[: -len(EXTENSION)]
    f_in.close()

    f_out = open(n_out, "ab")
    f_out.write(output)
    f_out.close()
    print("[INFO] Done")
