#pragma once

#include <cmath>
#include <iostream>

#include "utils.hpp"

struct BealeTrait
{   
    using Type = std::pair<float, float>;  
    static Type create()
    {
        return { random(-4.5F, 4.5F), random(-4.5F, 4.5F) };
    }
    static float fitness(const Type& type)
    {
        auto [x, y] = type;
        return std::pow(1.5F - x + x * y, 2) + std::pow(2.25F - x + x * y * y, 2) + std::pow(2.625F - x + x * y * y * y, 2);
    }
    static void mutate(Type& type)
    {
        type.first *= random(0.F, 2.F);
        type.second *= random(0.F, 2.F);
        clamp(type.first, -4.5F, 4.5F);
        clamp(type.second, -4.5F, 4.5F);
    }
    static Type crossover(const Type& pa, const Type& pb)
    {
        if (rand() % 2 == 0)
            return { pa.first, pb.second };
        else
            return { pb.first, pa.second }; 
    }
    static void print(const Type& type)
    {
        std::cout << "{" << type.first << ',' << type.second << "}\n";
    }
};