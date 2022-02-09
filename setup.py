from setuptools import setup

url = ""
version = "0.1.0"
readme = open('README.md').read()

setup(
    name="mouse_embryo_labeller",
    packages=["mouse_embryo_labeller"],
    version=version,
    description="A tool for tracking mouse embryo cell nuclei over time using data derived from 4d microscopy.",
    long_description=readme,
    include_package_data=True,
    author="Aaron Watters",
    author_email="awatters@flatironinstitute.org",
    url=url,
    install_requires=[
        "jp_doodle",
        "jp_proxy_widget",
        "scikit-image",
        ],
    scripts=[
        "bin/roi_from_tiff"
    ],
    # dependancy links doesn't work apparently because you need a tag.
    #dependency_links=[
    #    'https://github.com/AaronWatters/jp_doodle/tarball/master',
    #    ],
    license="MIT"
)
