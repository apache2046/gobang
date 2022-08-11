from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
import socket, struct

def ClientWithTimeout(address, authkey, timeout):

    with socket.socket(socket.AF_INET) as s:
        s.setblocking(True)
        s.connect(address)

        # We'd like to call s.settimeout(timeout) here, but that won't work.

        # Instead, prepare a C "struct timeval" to specify timeout. Note that
        # these field sizes may differ by platform.
        seconds = int(timeout)
        microseconds = int((timeout - seconds) * 1e6)
        timeval = struct.pack("@LL", seconds, microseconds)

        # And then set the SO_RCVTIMEO (receive timeout) option with this.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeval)

        # Now create the connection as normal.
        c = Connection(s.detach())

    # The following code will now fail if a socket timeout occurs.

    answer_challenge(c, authkey)
    deliver_challenge(c, authkey)

    return c
