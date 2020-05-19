import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quanttools", # Replace with your own username
    version="0.0.0",
    author="Carl Che",
    author_email="",
    description="A quantitative finance/ data science tool package",
    install_requires=['numpy', 'scipy', 'matplotlib'],

    url="https://github.com/carlche15/quanttools.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
