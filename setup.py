from cx_Freeze import setup, Executable

setup(name="Simple object detection software",
version="0.1",
description="This software detects object in realtime",
executables=[Executable("main.py")]
)