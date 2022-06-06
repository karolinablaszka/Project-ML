#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Karolina Blaszka & Paula Koralewska",
    author_email='kblaszka@edu.cdv.pl',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Projekt_ML",
    entry_points={
        'console_scripts': [
            'project_ML=project_ML.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='project_ML',
    name='project_ML',
    packages=find_packages(include=['project_ML', 'project_ML.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/karolinablaszka/project_ML',
    version='3.8',
    zip_safe=False,
)
