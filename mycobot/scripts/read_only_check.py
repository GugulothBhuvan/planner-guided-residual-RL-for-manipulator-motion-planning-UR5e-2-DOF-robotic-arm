"""Read-only MyCobot 280 check: connect, read joint angles and coords. Sends NO motion command.

Usage: python read_only_check.py [COM13] [115200]
Close myBlockly / any other program holding the port first.
"""
import sys
import time

from pymycobot import MyCobot280

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

mc = MyCobot280(PORT, BAUD)
time.sleep(0.5)

for i in range(5):
    angles = mc.get_angles()
    coords = mc.get_coords()
    print(f"{i}: angles={angles} coords={coords}")
    time.sleep(0.3)

for name in ("get_system_version", "get_robot_version", "get_basic_version", "get_robot_status"):
    fn = getattr(mc, name, None)
    if callable(fn):
        try:
            print(name, "->", fn())
        except Exception as exc:
            print(name, "failed:", exc)
