from setuptools import setup, find_packages

# Đọc nội dung từ tệp requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='RemoveNoiseSpeech_WaveUnet_VietAE',
    packages=find_packages(),
    install_requires=requirements,
)