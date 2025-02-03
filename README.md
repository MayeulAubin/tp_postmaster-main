# Convection simulation

This repo is taken from a lecture from CExA on High Performance Computing. 

## Fortran version

### How to compile

```bash
gfortran -ffree-line-length-512 -O3 src/convection.f90 -o convection.out
```

### How to execute

```bash
./convection.out
```

## C++ version

### How to compile

```bash
g++ -std=c++17 -O3 src/convection.cpp -o convection.out
```

### How to execute

```bash
./convection.out
```

## Python version

### How to execute

```bash
python3 src/convection.py
```

## How to read the output files with ParaView

We will use ParaView for 3d rendering of the output data. Although the simulation variables are defined as cell data, we will treat them as point data corresponding to the cell centers of the mesh within ParaView.

See section `Displaying data as structured grid` in <https://www.paraview.org/Wiki/ParaView/Data_formats>.
