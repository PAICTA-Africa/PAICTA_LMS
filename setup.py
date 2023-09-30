from setuptools import setup

setup(
    name='grade12_calculus_qa_model',
    version='1.0',
    description='Question and Answer Model for Grade 12 Calculus',
    packages=['model'],
    install_requires=[
        'tensorflow',
        'transformers',
        'torch',
        'numpy',
        'scikit-learn',
        'nltk',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
