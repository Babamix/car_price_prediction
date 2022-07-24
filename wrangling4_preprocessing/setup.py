import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = "car-price-prediction-preprocessing",
    version = "0.0.1",
    author = "Babamix",
    autor_email = "bagus.emeraldo@gmail.com",
    description = "simple repo for exercise",
    long_description = "...",
    long_description_content_type = "text/markdown",
    url = "",
    packages = setuptools.find_packages(),
    classifier = ["Programming Language :: Python :: 3"],
    install_requires = [
        "matplotlib",
        "numpy",
        "pandas == 1.1.4",
        "scikit-learn == 0.22", #(>= 3.5)
        "seaborn == 0.11.0"],
    python_requires = ">=3.7"
)