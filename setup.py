from setuptools import setup, find_packages

setup(
    name='fcpflow',  # Replace with your package name
    version='0.1.0',  # Initial version
    author='Weijie Xia',
    author_email='xiaweijie1996@gmail.com',
    description='A Flow-Based Model for Conditional and Probabilistic Electricity Consumption Profile Generation and Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Use 'text/markdown' if README.md is in Markdown
    url='https://github.com/xiaweijie1996/Full-Convolutional-Profile-Flow.git',  # URL of your project's repository
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=[
        'torch==2.2.2',
        'numpy==1.23.5',
        'matplotlib==3.7.0',
        'scikit-learn==1.2.2',
        'scipy==1.10.1',
        'huggingface-hub==0.23.0',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose a license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  # Minimum Python version required
)