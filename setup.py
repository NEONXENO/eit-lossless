from setuptools import setup, find_packages

setup(
    name="eit-lossless",
    version="1.0.0",
    author="NEO-SO + Grok AI (xAI)",
    description="Embedding Inactivation Technique - 10x faster infinite context for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NEONXENO/eit-lossless",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy",
        "matplotlib",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8+",
        "Programming Language :: Python :: 3.9+",
        "Programming Language :: Python :: 3.10+",
        "Programming Language :: Python :: 3.11+",
    ],
    python_requires=