from setuptools import setup, find_packages

setup(
    name="expai_lens",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],
    entry_points={"console_scripts": ["expai_lens=expai_lens.cli:run_dashboard"]},
)