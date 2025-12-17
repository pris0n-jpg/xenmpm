from setuptools import setup, find_packages
import glob

assets = glob.glob('./xengym/assets/**/*', recursive=True)
assets = [i.replace('./xengym/', '') for i in assets]

setup(
    name="xengym",
    version="v0.2.0",
    author="liujin",
    author_email="liuyvjin@qq.com",
    description="Xense Simulator",
    setup_requires=['cypack[build]'],
#    cypack=True,
    packages=find_packages(),
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    package_data={
        'xengym': assets,
    },
    entry_points={
        'console_scripts': [
            'xengym-demo = xengym.main:main',
        ],
    },
    install_requires=[
        'numpy<=1.26.4',
    ]
)
