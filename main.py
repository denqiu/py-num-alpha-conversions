import os

files_directory = os.path.join(os.path.dirname(__file__), 'files')
if not os.path.exists(files_directory):
    os.mkdir(files_directory)