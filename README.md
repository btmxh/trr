# trr

# Week 1: Using Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP)

**Run instructions**:

```sh
# Current working directory should be trr/

# Install dependencies
pip install -r ga-tsp/requirements.txt

# Run the script for the Western Sahara dataset
python ga-tsp/main.py

# Run the script for arbitrary dataset at path [PATH]
python ga-tsp/main.py [PATH]
```

TSP datasets taken from https://www.math.uwaterloo.ca/tsp/data/index.html

# Week 5: 
All example functions are taken from here: [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
## Using Differential Evolution to optimize some test functions

```sh
# Current working directory should be trr/

# Install dependencies
pip install -r de/requirements.txt

# Run the script for the Western Sahara dataset
python de/main.py
```
## C++ implementation of GA with example test functions

### Requirements:

* `CMake 3.20` or higher
* Functional build system such as `Unix Makefiles, Ninja` or `MSVC`

```sh
# Current working directory should be trr/
cd ga-test-func
cmake -S . -B build
cmake --build build

# Run the test
cd build
./ga sphere <num_of_dimensions> # or "ga beale"
```
