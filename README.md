# TRCD Data Processing

This library provides functionality for cleaning, reorganizing, fitting, and exporting data produced in time-resolved circular dichroism (TRCD) experiments.

# Background

Molecules absorb light differently depending on properties of both the molecule (atomic/electronic structure, environment, etc) and the light (wavelength/color/frequency, intensity, polarization, etc). The polarization of a beam of light tells you how the electromagnetic field of the beam oscillates. This oscillation can take a variety of forms, one of which is circular i.e. the electric field vector rotates in a circle perpendicular to the direction that the beam moves. 

There are two ways to go around a circle (clockwise or counterclockwise), and it turns out that some molecules absorb circularly polarized light different depending on whether the polarization is rotating one way or the other. Furthermore, some molecules only absorb these two polarizations differently for a short time after some kind of event has occurred. By triggering this kind of event and measuring the difference in absorbed light afterwards, we can examine some properties of the molecules.

This library provides functionality for cleaning, reorganizing, fitting, and exporting the results of these measurements.

# Caveats

* This library is under active development, so some functionality may be broken at any given time
* The design of this library is strongly coupled to our particular experimental apparatus, and thus expects raw data to be formatted a certain way (i.e. the way that our experiment generates it). 

# License

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
$ python -m trcdproc
```

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and a custom project template.

