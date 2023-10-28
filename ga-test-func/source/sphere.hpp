#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.hpp"

class SphereTrait
{
    static std::size_t& n()
    {
        static std::size_t res;
        return res;
    }
    public:
        static void init(std::size_t n)
        {
            std::srand(std::time(0));
            SphereTrait::n() = n;
        }
        using Type = std::vector<float>;

        static Type create()
        {
            std::vector<float> res;
            while (res.size() < n())
                res.push_back(random(0, 1000));
            return std::move(res);
        }
        static float fitness(const Type& type)
        {
            float res = 0.F;
            for (const auto& i : type)
            {
                res += i * i;
            }
            return res;
        }
        static void mutate(Type& type)
        {
            int j = std::rand() % n();
            type[j] /= 2.F;
        }
        static Type crossover(const Type& pa, const Type& pb)
        {
            int slice_start = std::rand() % n();
            Type res = pa;
            Type res2 = pb;
            std::sort(res2.rbegin(), res2.rend());
            std::sort(res.begin(), res.end());
            for (int i = slice_start; i < n(); ++i)
                res[i] = res2[i];
            return std::move(res);
        }
        static void print(const Type& type)
        {
            std::cout << "{";
            for (int i = 0; i < type.size(); ++i)
                std::cout << type[i] << ( i == type.size() - 1 ? "" : ",");
            std::cout << "}\n";
        }
};