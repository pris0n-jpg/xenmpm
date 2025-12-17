from pathlib import Path
PROJ_DIR = Path(__file__).resolve().parent

import platform
SYSTEM = platform.system().lower()
MACHINE = platform.machine().lower()

from xensesdk.xenseInterface.XenseSensor import Sensor as Sensor
from xensesdk.omni.widgets import ExampleView as ExampleView