#include <cstdio>
#include <cstring>
#include <iomanip>
#include <string>

#include "problem.hpp"
#include "sphere.hpp"
#include "beale.hpp"

int main(int argc, char* argv[])
{
    std::srand(std::time(0));
    std::size_t pop_size, cx_cnt, gen_cnt;
    float mut_rate;
    std::cout << std::fixed << std::setprecision(6);
    freopen("config.txt", "r", stdin);
    std::cin >> pop_size >> cx_cnt >> gen_cnt >> mut_rate;

    if (argc >= 3 && std::strcmp(argv[1], "sphere") == 0)
    {
        std::size_t n = std::stoi(std::string(argv[2]));
        SphereTrait::init(n);
        Problem<SphereTrait> pop{pop_size, cx_cnt, gen_cnt, mut_rate, true};
        pop.run();
    }
    else if (argc >= 2 && std::strcmp(argv[1], "beale") == 0)
    {
        Problem<BealeTrait>(pop_size, cx_cnt, gen_cnt, mut_rate, true).run();
    }
    else
        std::cout << "Unrecognized test" << std::endl;
    return -1;
}