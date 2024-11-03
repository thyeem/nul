import setuptools

setuptools.setup(
    name="nul-lm",
    version="0.0.0",
    description="nul",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="",
    author="Francis Lim",
    author_email="thyeem@gmail.com",
    license="MIT",
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="nul",
    packages=setuptools.find_packages(),
    install_requires=[],
    python_requires=">=3.6",
)
