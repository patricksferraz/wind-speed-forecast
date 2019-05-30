__author__ = "Ismail Akkila, Adapted by Patrick Ferraz"
__credits__ = ["Ismail Akkila", "Patrick Ferraz"]
__maintainer__ = "Patrick Ferraz"
__email__ = "patrick.ferraz@outlook.com"

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import zlib
import base64


# Our Encryption Function
def encrypt_blob(blob, public_key):
    # Import the Public Key and use for encryption using PKCS1_OAEP
    rsa_key = RSA.importKey(public_key)
    rsa_key = PKCS1_OAEP.new(rsa_key)

    # compress the data first
    blob = zlib.compress(blob)

    # In determining the chunk size, determine the private key length used in
    # bytes
    # and subtract 42 bytes (when using PKCS1_OAEP). The data will be in
    # encrypted
    # in chunks
    chunk_size = 470
    offset = 0
    end_loop = False
    encrypted = bytes()

    while not end_loop:
        # The chunk
        chunk = blob[offset : offset + chunk_size]

        # If the data chunk is less then the chunk size, then we need to add
        # padding with " ". This indicates the we reached the end of the file
        # so we end loop here
        if len(chunk) % chunk_size != 0:
            end_loop = True
            chunk += bytes(" " * (chunk_size - len(chunk)), encoding="utf-8")

        # Append the encrypted chunk to the overall encrypted file
        encrypted += rsa_key.encrypt(chunk)

        # Increase the offset by chunk size
        offset += chunk_size

    # Base 64 encode the encrypted file
    return base64.b64encode(encrypted)
