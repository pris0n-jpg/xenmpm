from setuptools import setup, find_packages
from cypack.build import _conf
import os

# _conf["keep_modules"] = [
#     "ti_kernel.py",
#     "cuda_kernel.py"
# ]

_conf["exclude"] = [
    "xensesdk/examples/*.py"
]

def detectMachineType():
    if os.path.exists("/etc/nv_tegra_release"):
        return "jetson"

    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "orange pi 5" in model:
                return "rk3588"
            elif "rdk x5" in model:
                return "rdk"
            elif "3576" in model:
                return "rk3576"
    return "x86"                

install_requires =[
        "cypack",
        'numpy<=1.26.4',
        'opencv-python==4.10.0.84',
        'PyOpenGL==3.1.7',
        'assimp-py==1.0.7',
        'pillow==10.2.0',
        'PySide6',
        'cryptography==43.0.3',
        'PyYAML==6.0.2',
        'qtpy',
        'h5py',
        'av==13.1.0',
        'scipy==1.13.1',
        "lz4",
        "pyzmq==24.0.1",
        "psutil>=5.9.0",
        'wmi; platform_system=="Windows"',     # 只在 Windows 安装
        'pyudev; platform_system=="Linux"',     # 只在 linux 安装
    ]

if detectMachineType() not in ["rk3588", "rdk", "rk3576"]:
    install_requires.append('onnxruntime-gpu==1.18.0')

setup(
    name="xensesdk",
    version="1.2.5",
    description="xense SDK for everything supported",
    author="Hongzhan Yu",
    author_email="hongzhanyu@xense",
    packages=find_packages(),
    setup_requires=['cypack[build]'],
    cypack=True,
    include_package_data=True,
    install_requires = install_requires,
    data_files=[
        ('xensesdk/lib', [
            'xensesdk/lib/CvCameraIndex.dll', 
            'xensesdk/lib/arm_64/libXenseBase.so', 
            'xensesdk/lib/arm_64/XenseWrapper.so',
            'xensesdk/lib/linux_64/libXenseBase.so',
            'xensesdk/lib/linux_64/XenseWrapper.so',
            'xensesdk/lib/win_x86/CLib.dll',
            'xensesdk/lib/win_x86/FRWLib.dll', 
        ]),  # 明确指定DLL文件
    ],
    package_data={
    'xensesdk.omni': ['assets/**/*.*', 'assets/*.*'],
    'xensesdk.ezgl': [
        'resources/textures/**/*.*',
        'resources/textures/*.*',
    ],
    'xensesdk': [
                "examples/example_force.py",
                "examples/example_marker_detect.py",
                "examples/example_depth.py",
                "examples/example_data_processing.py",
                "examples/example_record_data.py",
                "examples/example_finger_depth.py",
                "examples/example_remote_data.py"
                ],
    'xensesdk.xenseInterface': ['guiConfig/**/*.*', 'guiConfig/*.*'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'xense_demo=xensesdk.examples.example_depth:main',
        ],
    },
)
