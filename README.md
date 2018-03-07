====================
# TRCD Data Processing
====================

This is a flibrary for handling and processing TRCD data stored in HDF5 files.


* Free software: MIT license


# Usage

Install the dependencies
```bash
$ pip install -r requirements.txt
```

Run all of the tests
```bash
$ tox
```

Run the tests just covering the functionality of the code i.e. not the style or the formatting
```bash
$ tox -e py36
```

Check formatting and style
```bash
$ tox -e flake8
```

Run the project as a module via
```bash
$ python -m trcd_data_processing
```

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and a custom project template.

